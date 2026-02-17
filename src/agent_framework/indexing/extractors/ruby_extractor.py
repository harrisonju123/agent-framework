"""Ruby symbol extractor using regex-based line scanning with depth tracking."""

import re

from agent_framework.indexing.extractors.base import BaseExtractor
from agent_framework.indexing.models import SymbolEntry, SymbolKind

RE_CLASS = re.compile(r'^\s*class\s+([A-Z]\w*(?:::\w+)*)(?:\s*<\s*(\S+))?')
RE_MODULE = re.compile(r'^\s*module\s+([A-Z]\w*(?:::\w+)*)')
RE_METHOD = re.compile(r'^\s*def\s+(self\.)?(\w+[?!=]?)(?:\s*[\(;]|\s*$)')
RE_PRIVATE = re.compile(r'^\s*(private|protected)\s*$')
RE_END = re.compile(r'^\s*end\b')

# Patterns that open a block (increment depth on `end`)
RE_BLOCK_OPENER = re.compile(
    r'^\s*(?:if|unless|while|until|case|begin|for)\b'
)
RE_DO_BLOCK = re.compile(r'\bdo\s*(\|[^|]*\|)?\s*$')
RE_CLASS_SELF = re.compile(r'^\s*class\s*<<\s*self\b')


class RubyExtractor(BaseExtractor):

    def extract_symbols(self, file_path: str, source: str) -> list[SymbolEntry]:
        if not source.strip():
            return []

        lines = source.splitlines()
        symbols: list[SymbolEntry] = []
        depth = 0
        # Stack: (name, kind, entry_depth, is_private_zone)
        scope_stack: list[tuple[str, SymbolKind, int, bool]] = []
        is_private = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip blank and comment-only lines for block tracking
            if not stripped or stripped.startswith("#"):
                continue

            # class << self — synthetic scope
            if RE_CLASS_SELF.match(line):
                depth += 1
                scope_stack.append(("<<self", SymbolKind.CLASS, depth, is_private))
                continue

            # Module
            m = RE_MODULE.match(line)
            if m:
                mod_name = m.group(1)
                depth += 1
                docstring = self._extract_line_comments(lines, i, "#")
                parent = self._current_parent(scope_stack)
                symbols.append(SymbolEntry(
                    name=mod_name,
                    kind=SymbolKind.MODULE,
                    file_path=file_path,
                    line=i + 1,
                    signature=f"module {mod_name}",
                    docstring=self._truncate_docstring(docstring) if docstring else None,
                    parent=parent,
                ))
                scope_stack.append((mod_name, SymbolKind.MODULE, depth, False))
                is_private = False
                continue

            # Class
            m = RE_CLASS.match(line)
            if m:
                cls_name = m.group(1)
                base = m.group(2)
                depth += 1
                sig = f"class {cls_name} < {base}" if base else f"class {cls_name}"
                docstring = self._extract_line_comments(lines, i, "#")
                parent = self._current_parent(scope_stack)
                symbols.append(SymbolEntry(
                    name=cls_name,
                    kind=SymbolKind.CLASS,
                    file_path=file_path,
                    line=i + 1,
                    signature=sig,
                    docstring=self._truncate_docstring(docstring) if docstring else None,
                    parent=parent,
                ))
                scope_stack.append((cls_name, SymbolKind.CLASS, depth, False))
                is_private = False
                continue

            # Private/protected marker
            if RE_PRIVATE.match(line):
                is_private = True
                continue

            # Method definition
            m = RE_METHOD.match(line)
            if m:
                is_self_method = m.group(1) is not None
                method_name = m.group(2)
                depth += 1

                if not is_private:
                    kind = SymbolKind.METHOD if scope_stack else SymbolKind.FUNCTION
                    parent = self._current_parent(scope_stack)
                    # self. methods or methods inside class << self are class-level
                    prefix = "self." if is_self_method else ""
                    sig = f"def {prefix}{method_name}"
                    docstring = self._extract_line_comments(lines, i, "#")
                    symbols.append(SymbolEntry(
                        name=f"{prefix}{method_name}",
                        kind=kind,
                        file_path=file_path,
                        line=i + 1,
                        signature=sig,
                        docstring=self._truncate_docstring(docstring) if docstring else None,
                        parent=parent,
                    ))
                continue

            # end — pop scope if depth matches
            if RE_END.match(line):
                if scope_stack and scope_stack[-1][2] == depth:
                    _, _, _, _ = scope_stack.pop()
                    # Restore private state from parent scope
                    is_private = scope_stack[-1][3] if scope_stack else False
                depth = max(0, depth - 1)
                continue

            # Other block openers that need depth tracking
            if RE_BLOCK_OPENER.match(line) or RE_DO_BLOCK.search(line):
                depth += 1

        return symbols

    def _current_parent(
        self, scope_stack: list[tuple[str, SymbolKind, int, bool]]
    ) -> str | None:
        """Return the nearest real (non-synthetic) parent name."""
        for name, _, _, _ in reversed(scope_stack):
            if name != "<<self":
                return name
        return None
