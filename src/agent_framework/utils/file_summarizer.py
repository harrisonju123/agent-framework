"""Lightweight structural summaries for the read cache.

Generates one-line summaries (line count + key symbols) so downstream
agents can decide whether to re-read a file without opening it.
"""

import ast
import re

_MAX_FILE_BYTES = 100_000
_BINARY_CHECK_BYTES = 512

# Extension groups
_PYTHON_EXTS = {".py"}
_JS_TS_EXTS = {".ts", ".tsx", ".js", ".jsx"}
_CONFIG_EXTS = {".yaml", ".yml", ".json", ".toml"}
_DOC_EXTS = {".md", ".rst", ".txt"}

# Regex for JS/TS exported symbols
_JS_EXPORT_RE = re.compile(
    r"export\s+(?:default\s+)?(?:async\s+)?"
    r"(?:function|class|interface|type|const|let|var|enum)\s+"
    r"(\w+)",
)


def summarize_file(file_path: str, max_length: int = 120) -> str:
    """Return a one-line structural summary of a file.

    Returns empty string on any error (missing, unreadable, etc.).
    """
    try:
        with open(file_path, "rb") as f:
            head = f.read(_BINARY_CHECK_BYTES)
            if b"\x00" in head:
                return "Binary file"
            f.seek(0)
            raw = f.read(_MAX_FILE_BYTES)
    except OSError:
        return ""

    text = raw.decode("utf-8", errors="replace")

    lines = text.splitlines()
    line_count = len(lines)
    if line_count == 0:
        return "0L; empty"

    ext = _get_extension(file_path)

    if ext in _PYTHON_EXTS:
        summary = _summarize_python(text, line_count)
    elif ext in _JS_TS_EXTS:
        summary = _summarize_js_ts(text, line_count)
    elif ext in _CONFIG_EXTS:
        summary = f"{line_count}L; config"
    elif ext in _DOC_EXTS:
        summary = f"{line_count}L; docs"
    else:
        summary = f"{line_count}L"

    if len(summary) > max_length:
        summary = summary[: max_length - 3] + "..."
    return summary


def _get_extension(file_path: str) -> str:
    dot = file_path.rfind(".")
    return file_path[dot:].lower() if dot != -1 else ""


def _summarize_python(text: str, line_count: int) -> str:
    """Extract top-level classes and functions via AST, fall back to line count."""
    classes: list[str] = []
    funcs: list[str] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return f"{line_count}L"

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node.name)

    parts = [f"{line_count}L"]
    if classes:
        parts.append(f"classes: {', '.join(classes)}")
    if funcs:
        parts.append(f"funcs: {', '.join(funcs)}")
    return "; ".join(parts)


def _summarize_js_ts(text: str, line_count: int) -> str:
    """Extract exported symbols via regex."""
    exports = _JS_EXPORT_RE.findall(text)
    parts = [f"{line_count}L"]
    if exports:
        parts.append(f"exports: {', '.join(exports)}")
    return "; ".join(parts)
