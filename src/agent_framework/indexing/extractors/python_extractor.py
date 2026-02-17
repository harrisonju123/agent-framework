"""Python symbol extractor using the ast module."""

import ast

from agent_framework.indexing.extractors.base import BaseExtractor
from agent_framework.indexing.models import SymbolEntry, SymbolKind


class PythonExtractor(BaseExtractor):

    def extract_symbols(self, file_path: str, source: str) -> list[SymbolEntry]:
        if not source.strip():
            return []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        symbols: list[SymbolEntry] = []
        self._walk(tree, file_path, symbols, parent=None)
        return symbols

    def _walk(
        self,
        node: ast.AST,
        file_path: str,
        symbols: list[SymbolEntry],
        parent: str | None,
    ) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                self._extract_class(child, file_path, symbols, parent)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if parent is None:
                    self._extract_function(child, file_path, symbols)

    def _extract_class(
        self,
        node: ast.ClassDef,
        file_path: str,
        symbols: list[SymbolEntry],
        parent: str | None,
    ) -> None:
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        sig = f"class {node.name}({bases})" if bases else f"class {node.name}"
        docstring = ast.get_docstring(node)

        symbols.append(SymbolEntry(
            name=node.name,
            kind=SymbolKind.CLASS,
            file_path=file_path,
            line=node.lineno,
            signature=sig,
            docstring=self._truncate_docstring(docstring) if docstring else None,
            parent=parent,
        ))

        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                self._extract_class(child, file_path, symbols, parent=node.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._extract_method(child, file_path, symbols, parent=node.name)

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: str,
        symbols: list[SymbolEntry],
    ) -> None:
        if node.name.startswith("_"):
            return

        prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        sig = f"{prefix}def {node.name}({self._format_args(node.args)})"
        if node.returns:
            sig += f" -> {ast.unparse(node.returns)}"

        docstring = ast.get_docstring(node)
        symbols.append(SymbolEntry(
            name=node.name,
            kind=SymbolKind.FUNCTION,
            file_path=file_path,
            line=node.lineno,
            signature=sig,
            docstring=self._truncate_docstring(docstring) if docstring else None,
        ))

    def _extract_method(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: str,
        symbols: list[SymbolEntry],
        parent: str,
    ) -> None:
        # Skip private methods except __init__
        if node.name.startswith("_") and node.name != "__init__":
            return

        prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        args_str = self._format_args(node.args, skip_self=True)
        sig = f"{prefix}def {node.name}({args_str})"
        if node.returns:
            sig += f" -> {ast.unparse(node.returns)}"

        docstring = ast.get_docstring(node)
        symbols.append(SymbolEntry(
            name=node.name,
            kind=SymbolKind.METHOD,
            file_path=file_path,
            line=node.lineno,
            signature=sig,
            docstring=self._truncate_docstring(docstring) if docstring else None,
            parent=parent,
        ))

    def _format_args(self, args: ast.arguments, skip_self: bool = False) -> str:
        parts: list[str] = []

        # Positional-only args (before the / separator)
        for arg in args.posonlyargs:
            s = arg.arg
            if arg.annotation:
                s += f": {ast.unparse(arg.annotation)}"
            parts.append(s)

        if args.posonlyargs:
            parts.append("/")

        # Positional args (includes self/cls)
        num_defaults = len(args.defaults)
        num_args = len(args.args)
        non_default_count = num_args - num_defaults

        for i, arg in enumerate(args.args):
            if skip_self and i == 0 and arg.arg in ("self", "cls"):
                continue
            s = arg.arg
            if arg.annotation:
                s += f": {ast.unparse(arg.annotation)}"
            if i >= non_default_count:
                default = args.defaults[i - non_default_count]
                s += f" = {ast.unparse(default)}"
            parts.append(s)

        # *args
        if args.vararg:
            s = f"*{args.vararg.arg}"
            if args.vararg.annotation:
                s += f": {ast.unparse(args.vararg.annotation)}"
            parts.append(s)
        elif args.kwonlyargs:
            parts.append("*")

        # Keyword-only args
        for i, arg in enumerate(args.kwonlyargs):
            s = arg.arg
            if arg.annotation:
                s += f": {ast.unparse(arg.annotation)}"
            if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
                s += f" = {ast.unparse(args.kw_defaults[i])}"
            parts.append(s)

        # **kwargs
        if args.kwarg:
            s = f"**{args.kwarg.arg}"
            if args.kwarg.annotation:
                s += f": {ast.unparse(args.kwarg.annotation)}"
            parts.append(s)

        return ", ".join(parts)
