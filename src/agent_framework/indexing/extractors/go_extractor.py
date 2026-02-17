"""Go symbol extractor using regex-based line scanning."""

import re

from agent_framework.indexing.extractors.base import BaseExtractor
from agent_framework.indexing.models import SymbolEntry, SymbolKind

# Only exported symbols (uppercase first letter)
RE_FUNC = re.compile(
    r'^func\s+([A-Z]\w*)(?:\[[\w\s,]+\])?\s*\('
)
RE_METHOD = re.compile(
    r'^func\s+\((\w+)\s+(\*?)(\w+)(?:\[[\w\s,]+\])?\)\s+([A-Z]\w*)\s*\('
)
RE_TYPE = re.compile(
    r'^type\s+([A-Z]\w*)(?:\[[\w\s,]+\])?\s+(struct|interface)\s*\{?'
)


class GoExtractor(BaseExtractor):

    def extract_symbols(self, file_path: str, source: str) -> list[SymbolEntry]:
        if not source.strip():
            return []

        lines = source.splitlines()
        symbols: list[SymbolEntry] = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Try method first (more specific pattern)
            m = RE_METHOD.match(line)
            if m:
                recv_name, ptr, recv_type, method_name = m.groups()
                full_sig = self._join_multiline_signature(lines, i)
                sig = self._build_signature(full_sig)
                docstring = self._extract_line_comments(lines, i, "//")
                symbols.append(SymbolEntry(
                    name=method_name,
                    kind=SymbolKind.METHOD,
                    file_path=file_path,
                    line=i + 1,
                    signature=sig,
                    docstring=self._truncate_docstring(docstring) if docstring else None,
                    parent=recv_type,
                ))
                i += 1
                continue

            # Top-level function
            m = RE_FUNC.match(line)
            if m:
                func_name = m.group(1)
                full_sig = self._join_multiline_signature(lines, i)
                sig = self._build_signature(full_sig)
                docstring = self._extract_line_comments(lines, i, "//")
                symbols.append(SymbolEntry(
                    name=func_name,
                    kind=SymbolKind.FUNCTION,
                    file_path=file_path,
                    line=i + 1,
                    signature=sig,
                    docstring=self._truncate_docstring(docstring) if docstring else None,
                ))
                i += 1
                continue

            # Type declaration (struct/interface)
            m = RE_TYPE.match(line)
            if m:
                type_name = m.group(1)
                type_kind = m.group(2)
                kind = SymbolKind.STRUCT if type_kind == "struct" else SymbolKind.INTERFACE
                sig = f"type {type_name} {type_kind}"
                docstring = self._extract_line_comments(lines, i, "//")
                symbols.append(SymbolEntry(
                    name=type_name,
                    kind=kind,
                    file_path=file_path,
                    line=i + 1,
                    signature=sig,
                    docstring=self._truncate_docstring(docstring) if docstring else None,
                ))
                i += 1
                continue

            i += 1

        return symbols

    def _join_multiline_signature(self, lines: list[str], start: int) -> str:
        """Accumulate lines until parentheses balance, handling the full func signature."""
        accumulated = lines[start]
        depth = 0
        in_string = False
        string_char = ""

        # Scan character by character for paren balance
        for ch in accumulated:
            if in_string:
                if ch == string_char:
                    in_string = False
                continue
            if ch in ('"', '`'):
                in_string = True
                string_char = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1

        i = start + 1
        while depth > 0 and i < len(lines):
            next_line = lines[i].strip()
            accumulated += " " + next_line
            for ch in next_line:
                if in_string:
                    if ch == string_char:
                        in_string = False
                    continue
                if ch in ('"', '`'):
                    in_string = True
                    string_char = ch
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
            i += 1

        # Check for return type tuple on next line(s)
        if i < len(lines):
            rest = lines[i].strip()
            if rest.startswith("("):
                depth = 0
                for ch in rest:
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                accumulated += " " + rest
                i += 1
                while depth > 0 and i < len(lines):
                    next_line = lines[i].strip()
                    accumulated += " " + next_line
                    for ch in next_line:
                        if ch == "(":
                            depth += 1
                        elif ch == ")":
                            depth -= 1
                    i += 1

        return accumulated

    def _build_signature(self, raw: str) -> str:
        """Clean up a multiline signature into a single-line form, stripping the body."""
        # Remove everything after the opening brace of the function body
        sig = re.sub(r'\s*\{.*', '', raw)
        # Collapse whitespace
        sig = re.sub(r'\s+', ' ', sig).strip()
        return sig

