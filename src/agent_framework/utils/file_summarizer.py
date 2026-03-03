"""Lightweight structural summaries for the read cache.

Generates one-line summaries (line count + key symbols) so downstream
agents can decide whether to re-read a file without opening it.

Two summary levels:
- ``summarize_file()`` — compact one-liner for table rows (default <=120 chars)
- ``summarize_file_rich()`` — structural detail with method signatures, constants,
  imports, and anchor lines so LLMs can orient without re-reading the full file
"""

import ast
import re
from dataclasses import dataclass, field

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

# Regex for JS/TS function signatures
_JS_FUNC_SIG_RE = re.compile(
    r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)",
)

# Regex for top-level UPPER_CASE constants in Python
_PY_CONST_RE = re.compile(r"^([A-Z][A-Z0-9_]{2,})\s*=", re.MULTILINE)

# Regex for JS/TS top-level constants
_JS_CONST_RE = re.compile(r"^(?:export\s+)?const\s+([A-Z][A-Z0-9_]{2,})\s*=", re.MULTILINE)

# Go exported symbols
_GO_FUNC_RE = re.compile(r"^func\s+(?:\([^)]+\)\s+)?([A-Z]\w*)\s*\(([^)]*)\)", re.MULTILINE)
_GO_TYPE_RE = re.compile(r"^type\s+([A-Z]\w*)\s+(?:struct|interface)", re.MULTILINE)
_GO_CONST_RE = re.compile(r"^(?:const\s+)?([A-Z][A-Z0-9_]{2,})\s*=", re.MULTILINE)

_GO_EXTS = {".go"}


@dataclass
class RichSummary:
    """Structured summary with enough detail for LLMs to skip re-reading."""
    line_count: int
    classes: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    anchor_head: list[str] = field(default_factory=list)
    anchor_tail: list[str] = field(default_factory=list)

    def format(self, max_length: int = 400) -> str:
        """Render as a single-line summary for prompt injection."""
        parts = [f"{self.line_count}L"]
        if self.classes:
            parts.append(f"Classes: {', '.join(self.classes)}")
        if self.methods:
            parts.append(f"Methods: {', '.join(self.methods)}")
        if self.constants:
            parts.append(f"Constants: {', '.join(self.constants)}")
        if self.imports:
            parts.append(f"Imports: {', '.join(self.imports)}")

        result = ". ".join(parts)
        if len(result) > max_length:
            result = result[: max_length - 3] + "..."
        return result


def _read_file_text(file_path: str) -> tuple[str, list[str]] | None:
    """Read file and return (text, lines) or None on error/binary."""
    try:
        with open(file_path, "rb") as f:
            head = f.read(_BINARY_CHECK_BYTES)
            if b"\x00" in head:
                return None
            f.seek(0)
            raw = f.read(_MAX_FILE_BYTES)
    except OSError:
        return None

    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return text, lines


def summarize_file(file_path: str, max_length: int = 120) -> str:
    """Return a one-line structural summary of a file.

    Returns empty string on any error (missing, unreadable, etc.).
    """
    result = _read_file_text(file_path)
    if result is None:
        # Distinguish binary from missing/unreadable
        try:
            with open(file_path, "rb") as f:
                head = f.read(_BINARY_CHECK_BYTES)
                if b"\x00" in head:
                    return "Binary file"
        except OSError:
            pass
        return ""

    text, lines = result
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


def summarize_file_rich(file_path: str, max_length: int = 400) -> str:
    """Return a rich structural summary with signatures, constants, and anchors.

    Provides enough detail that an LLM can orient in the file without
    re-reading it. Target size: 200-400 chars per file.
    Returns empty string on any error.
    """
    result = _read_file_text(file_path)
    if result is None:
        return ""

    text, lines = result
    line_count = len(lines)
    if line_count == 0:
        return "0L; empty"

    ext = _get_extension(file_path)

    if ext in _PYTHON_EXTS:
        rich = _rich_python(text, lines)
    elif ext in _JS_TS_EXTS:
        rich = _rich_js_ts(text, lines)
    elif ext in _GO_EXTS:
        rich = _rich_go(text, lines)
    elif ext in _CONFIG_EXTS:
        return f"{line_count}L; config"
    elif ext in _DOC_EXTS:
        return f"{line_count}L; docs"
    else:
        return f"{line_count}L"

    rich.line_count = line_count
    return rich.format(max_length)


def _get_extension(file_path: str) -> str:
    dot = file_path.rfind(".")
    return file_path[dot:].lower() if dot != -1 else ""


def _extract_anchors(lines: list[str]) -> tuple[list[str], list[str]]:
    """Return first 3 and last 3 non-blank lines as orientation anchors."""
    non_blank = [ln.rstrip() for ln in lines if ln.strip()]
    head = non_blank[:3]
    tail = non_blank[-3:] if len(non_blank) > 3 else []
    return head, tail


def _format_py_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Format function name + parameter names (no type annotations for brevity)."""
    params = []
    for arg in node.args.args:
        if arg.arg != "self" and arg.arg != "cls":
            params.append(arg.arg)
    if params:
        return f"{node.name}({', '.join(params)})"
    return f"{node.name}()"


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


def _rich_python(text: str, lines: list[str]) -> RichSummary:
    """Build a rich summary for Python files via AST."""
    rich = RichSummary(line_count=len(lines))
    rich.anchor_head, rich.anchor_tail = _extract_anchors(lines)

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return rich

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            rich.classes.append(node.name)
            # Extract method signatures from the class body
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    rich.methods.append(_format_py_signature(item))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            rich.methods.append(_format_py_signature(node))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                rich.imports.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    rich.imports.append(alias.name)

    # Top-level UPPER_CASE constants via regex (AST makes this verbose)
    rich.constants = _PY_CONST_RE.findall(text)[:10]

    # Deduplicate imports (keep first 8)
    seen: set[str] = set()
    unique_imports: list[str] = []
    for imp in rich.imports:
        if imp not in seen:
            seen.add(imp)
            unique_imports.append(imp)
    rich.imports = unique_imports[:8]

    # Cap methods to keep summary compact
    rich.methods = rich.methods[:15]

    return rich


def _rich_js_ts(text: str, lines: list[str]) -> RichSummary:
    """Build a rich summary for JS/TS files via regex."""
    rich = RichSummary(line_count=len(lines))
    rich.anchor_head, rich.anchor_tail = _extract_anchors(lines)

    # Exported classes/interfaces/types (names only)
    rich.classes = _JS_EXPORT_RE.findall(text)[:10]

    # Function signatures with params
    for m in _JS_FUNC_SIG_RE.finditer(text):
        params = m.group(2).strip()
        if params:
            # Trim type annotations for brevity
            param_names = [p.strip().split(":")[0].split("=")[0].strip() for p in params.split(",")]
            rich.methods.append(f"{m.group(1)}({', '.join(param_names)})")
        else:
            rich.methods.append(f"{m.group(1)}()")
    rich.methods = rich.methods[:15]

    rich.constants = _JS_CONST_RE.findall(text)[:10]

    return rich


def _rich_go(text: str, lines: list[str]) -> RichSummary:
    """Build a rich summary for Go files via regex."""
    rich = RichSummary(line_count=len(lines))
    rich.anchor_head, rich.anchor_tail = _extract_anchors(lines)

    # Exported types (struct/interface)
    rich.classes = _GO_TYPE_RE.findall(text)[:10]

    # Exported function signatures
    for m in _GO_FUNC_RE.finditer(text):
        params = m.group(2).strip()
        if params:
            param_names = [p.strip().split(" ")[0] for p in params.split(",")]
            rich.methods.append(f"{m.group(1)}({', '.join(param_names)})")
        else:
            rich.methods.append(f"{m.group(1)}()")
    rich.methods = rich.methods[:15]

    rich.constants = _GO_CONST_RE.findall(text)[:10]

    return rich


def _summarize_js_ts(text: str, line_count: int) -> str:
    """Extract exported symbols via regex."""
    exports = _JS_EXPORT_RE.findall(text)
    parts = [f"{line_count}L"]
    if exports:
        parts.append(f"exports: {', '.join(exports)}")
    return "; ".join(parts)
