"""Tests for file_summarizer.summarize_file() and summarize_file_rich()."""

import os


from agent_framework.utils.file_summarizer import summarize_file, summarize_file_rich, RichSummary


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


class TestRichSummaryFormat:
    """RichSummary.format() output."""

    def test_format_includes_all_fields(self):
        rich = RichSummary(
            line_count=150,
            classes=["Agent", "Config"],
            methods=["__init__(workspace, config)", "run()"],
            constants=["MAX_RETRIES", "DEFAULT_TIMEOUT"],
            imports=["logging", "json"],
        )
        result = rich.format()
        assert "150L" in result
        assert "Classes: Agent, Config" in result
        assert "Methods: __init__(workspace, config), run()" in result
        assert "Constants: MAX_RETRIES, DEFAULT_TIMEOUT" in result
        assert "Imports: logging, json" in result

    def test_format_truncates_at_max_length(self):
        rich = RichSummary(
            line_count=100,
            classes=[f"Class{i}" for i in range(20)],
            methods=[f"method_{i}()" for i in range(20)],
        )
        result = rich.format(max_length=80)
        assert len(result) <= 80
        assert result.endswith("...")

    def test_format_empty_fields_omitted(self):
        rich = RichSummary(line_count=5)
        result = rich.format()
        assert result == "5L"
        assert "Classes:" not in result
        assert "Methods:" not in result


class TestSummarizeFileRichPython:
    """Rich summaries for Python files."""

    def test_includes_method_signatures(self, tmp_path):
        f = tmp_path / "agent.py"
        f.write_text(
            "import logging\nimport json\n\n"
            "MAX_RETRIES = 3\nDEFAULT_TIMEOUT = 30\n\n"
            "class Agent:\n"
            "    def __init__(self, workspace, config):\n        pass\n\n"
            "    def run(self, task):\n        pass\n\n"
            "def helper(x, y):\n    pass\n"
        )
        result = summarize_file_rich(str(f))
        assert "Classes: Agent" in result
        assert "Methods:" in result
        assert "__init__(workspace, config)" in result
        assert "run(task)" in result
        assert "helper(x, y)" in result
        assert "Constants: MAX_RETRIES, DEFAULT_TIMEOUT" in result
        assert "Imports:" in result

    def test_syntax_error_returns_line_count(self, tmp_path):
        f = tmp_path / "broken.py"
        f.write_text("def broken(\n")
        result = summarize_file_rich(str(f))
        assert "1L" in result

    def test_empty_python_file(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("")
        assert summarize_file_rich(str(f)) == "0L; empty"

    def test_missing_file_returns_empty(self):
        assert summarize_file_rich("/nonexistent/file.py") == ""

    def test_respects_max_length(self, tmp_path):
        f = tmp_path / "big.py"
        funcs = [f"def func_{i}(a, b, c): pass" for i in range(30)]
        f.write_text("\n".join(funcs) + "\n")
        result = summarize_file_rich(str(f), max_length=150)
        assert len(result) <= 150

    def test_self_cls_omitted_from_params(self, tmp_path):
        """self and cls should not appear in method signatures."""
        f = tmp_path / "example.py"
        f.write_text(
            "class Foo:\n"
            "    def bar(self, x, y):\n        pass\n"
            "    @classmethod\n"
            "    def create(cls, name):\n        pass\n"
        )
        result = summarize_file_rich(str(f))
        assert "bar(x, y)" in result
        assert "create(name)" in result
        assert "self" not in result
        assert "cls" not in result

    def test_imports_deduped_and_capped(self, tmp_path):
        f = tmp_path / "many_imports.py"
        imports = [f"import module_{i}" for i in range(15)]
        f.write_text("\n".join(imports) + "\n")
        result = summarize_file_rich(str(f))
        assert "Imports:" in result
        # Should be capped at 8 unique imports
        import_section = result.split("Imports: ")[1] if "Imports: " in result else ""
        import_count = len(import_section.split(", ")) if import_section else 0
        assert import_count <= 8


class TestSummarizeFileRichJsTs:
    """Rich summaries for JS/TS files."""

    def test_includes_function_signatures(self, tmp_path):
        f = tmp_path / "service.ts"
        f.write_text(
            "export const MAX_ITEMS = 100;\n"
            "export class UserService {}\n"
            "export function fetchUser(id: string, options: Options) {}\n"
            "function internalHelper(x) {}\n"
        )
        result = summarize_file_rich(str(f))
        assert "Classes:" in result
        assert "Methods:" in result
        assert "fetchUser(id, options)" in result
        assert "internalHelper(x)" in result
        assert "Constants: MAX_ITEMS" in result

    def test_empty_js_file(self, tmp_path):
        f = tmp_path / "empty.js"
        f.write_text("")
        assert summarize_file_rich(str(f)) == "0L; empty"


class TestSummarizeFileRichGo:
    """Rich summaries for Go files."""

    def test_exported_types_and_funcs(self, tmp_path):
        f = tmp_path / "handler.go"
        f.write_text(
            "package main\n\n"
            "type Handler struct {\n\tdb *DB\n}\n\n"
            "type Service interface {\n\tRun()\n}\n\n"
            "func NewHandler(db *DB, logger Logger) *Handler {\n\treturn nil\n}\n\n"
            "func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {\n}\n"
        )
        result = summarize_file_rich(str(f))
        assert "Classes: Handler, Service" in result
        assert "NewHandler(db, logger)" in result

    def test_go_constants(self, tmp_path):
        f = tmp_path / "consts.go"
        f.write_text(
            "package config\n\n"
            "const MAX_CONNECTIONS = 100\n"
            "const DEFAULT_PORT = 8080\n"
        )
        result = summarize_file_rich(str(f))
        assert "Constants:" in result
        assert "MAX_CONNECTIONS" in result


class TestSummarizeFileRichConfigDocs:
    """Rich summaries for config and doc files fall back to basic format."""

    def test_config_returns_basic(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("key: value\nother: 42\n")
        assert summarize_file_rich(str(f)) == "2L; config"

    def test_docs_returns_basic(self, tmp_path):
        f = tmp_path / "README.md"
        f.write_text("# Title\n\nSome text.\n")
        assert summarize_file_rich(str(f)) == "3L; docs"

    def test_unknown_extension_returns_line_count(self, tmp_path):
        f = tmp_path / "data.parquet"
        f.write_text("row1\nrow2\n")
        assert summarize_file_rich(str(f)) == "2L"
