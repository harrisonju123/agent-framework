"""Tests for detect_language_mismatches() — path confusion detection."""

from agent_framework.memory.tool_pattern_analyzer import (
    LanguageMismatch,
    detect_language_mismatches,
)


def _make_call(tool: str, **input_fields) -> dict:
    """Build a minimal tool_call event dict."""
    entry = {"event": "tool_call", "tool": tool}
    if input_fields:
        entry["input"] = input_fields
    return entry


class TestGlobMismatch:
    def test_go_glob_in_python_project(self):
        calls = [_make_call("Glob", pattern="**/*.go")]
        result = detect_language_mismatches(calls, "python")
        assert len(result) == 1
        assert result[0].searched_extension == ".go"
        assert result[0].searched_language == "go"
        assert result[0].project_language == "python"
        assert result[0].tool == "Glob"

    def test_multiple_extensions_in_single_glob(self):
        calls = [_make_call("Glob", pattern="**/*.go,**/*.rb")]
        result = detect_language_mismatches(calls, "python")
        assert len(result) == 2
        exts = {m.searched_extension for m in result}
        assert exts == {".go", ".rb"}


class TestGrepMismatch:
    def test_ts_grep_in_python_project(self):
        calls = [_make_call("Grep", pattern="import", glob="*.ts")]
        result = detect_language_mismatches(calls, "python")
        assert len(result) == 1
        assert result[0].searched_extension == ".ts"
        assert result[0].searched_language == "typescript"
        assert result[0].tool == "Grep"

    def test_grep_type_param_go_in_python_project(self):
        calls = [_make_call("Grep", pattern="func", type="go")]
        result = detect_language_mismatches(calls, "python")
        assert len(result) == 1
        assert result[0].searched_extension == ".go"
        assert result[0].searched_language == "go"

    def test_grep_type_param_correct_language(self):
        calls = [_make_call("Grep", pattern="def", type="py")]
        result = detect_language_mismatches(calls, "python")
        assert len(result) == 0

    def test_grep_type_param_unknown_type(self):
        """Unknown rg types (not in _RG_TYPE_TO_EXT) are ignored."""
        calls = [_make_call("Grep", pattern="fn", type="haskell")]
        result = detect_language_mismatches(calls, "python")
        assert len(result) == 0

    def test_grep_without_glob_no_match(self):
        calls = [_make_call("Grep", pattern="import")]
        result = detect_language_mismatches(calls, "python")
        assert len(result) == 0


class TestReadMismatch:
    def test_go_read_in_python_project(self):
        calls = [_make_call("Read", file_path="/repo/main.go")]
        result = detect_language_mismatches(calls, "python")
        assert len(result) == 1
        assert result[0].searched_extension == ".go"
        assert result[0].searched_language == "go"
        assert result[0].tool == "Read"

    def test_read_without_file_path(self):
        calls = [_make_call("Read")]
        result = detect_language_mismatches(calls, "python")
        assert len(result) == 0


class TestCorrectLanguage:
    def test_py_in_python_project(self):
        calls = [
            _make_call("Glob", pattern="**/*.py"),
            _make_call("Read", file_path="/repo/app.py"),
            _make_call("Grep", pattern="def", glob="*.py"),
        ]
        result = detect_language_mismatches(calls, "python")
        assert len(result) == 0

    def test_go_in_go_project(self):
        calls = [_make_call("Glob", pattern="**/*.go")]
        result = detect_language_mismatches(calls, "go")
        assert len(result) == 0

    def test_case_insensitive_project_language(self):
        """Index may store language as "Python" — should still match .py files."""
        calls = [_make_call("Glob", pattern="**/*.py")]
        result = detect_language_mismatches(calls, "Python")
        assert len(result) == 0

        calls = [_make_call("Glob", pattern="**/*.go")]
        result = detect_language_mismatches(calls, "Python")
        assert len(result) == 1


class TestAgnosticExtensions:
    def test_json_ignored(self):
        calls = [_make_call("Glob", pattern="**/*.json")]
        result = detect_language_mismatches(calls, "python")
        assert len(result) == 0

    def test_md_ignored(self):
        calls = [_make_call("Read", file_path="/repo/README.md")]
        result = detect_language_mismatches(calls, "python")
        assert len(result) == 0

    def test_yaml_ignored(self):
        calls = [_make_call("Glob", pattern="**/*.yaml")]
        result = detect_language_mismatches(calls, "go")
        assert len(result) == 0


class TestEmptyProjectLanguage:
    def test_empty_string(self):
        calls = [_make_call("Glob", pattern="**/*.go")]
        result = detect_language_mismatches(calls, "")
        assert len(result) == 0

    def test_empty_calls(self):
        result = detect_language_mismatches([], "python")
        assert len(result) == 0


class TestDeduplication:
    def test_same_ext_and_tool_deduped(self):
        calls = [
            _make_call("Glob", pattern="**/*.go"),
            _make_call("Glob", pattern="src/**/*.go"),
        ]
        result = detect_language_mismatches(calls, "python")
        assert len(result) == 1

    def test_same_ext_different_tool_not_deduped(self):
        calls = [
            _make_call("Glob", pattern="**/*.go"),
            _make_call("Read", file_path="/repo/main.go"),
        ]
        result = detect_language_mismatches(calls, "python")
        assert len(result) == 2


class TestCrossLanguage:
    def test_ts_in_go_project(self):
        calls = [_make_call("Grep", pattern="func", glob="*.ts")]
        result = detect_language_mismatches(calls, "go")
        assert len(result) == 1
        assert result[0].searched_language == "typescript"

    def test_ruby_in_typescript_project(self):
        calls = [_make_call("Read", file_path="/repo/app.rb")]
        result = detect_language_mismatches(calls, "typescript")
        assert len(result) == 1
        assert result[0].searched_language == "ruby"


class TestUnknownExtensions:
    def test_unknown_ext_ignored(self):
        """Extensions not in _EXT_TO_LANG are silently ignored."""
        calls = [_make_call("Glob", pattern="**/*.rs")]
        result = detect_language_mismatches(calls, "python")
        assert len(result) == 0

    def test_no_extension_ignored(self):
        calls = [_make_call("Read", file_path="/repo/Makefile")]
        result = detect_language_mismatches(calls, "python")
        assert len(result) == 0
