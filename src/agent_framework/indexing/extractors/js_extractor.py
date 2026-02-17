"""JavaScript/TypeScript symbol extractor using regex-based line scanning."""

import re

from agent_framework.indexing.extractors.base import BaseExtractor
from agent_framework.indexing.models import SymbolEntry, SymbolKind

RE_CLASS = re.compile(
    r'^\s*(?:export\s+(?:default\s+)?)?class\s+(\w+)(?:\s+extends\s+(\w+))?'
)
RE_FUNCTION = re.compile(
    r'^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+(\w+)\s*(?:<[^>]+>)?\s*\('
)
RE_ARROW = re.compile(
    r'^\s*(?:export\s+(?:default\s+)?)?(?:const|let|var)\s+(\w+).*?=\s*(?:async\s+)?\('
)
RE_INTERFACE = re.compile(
    r'^\s*(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+(\w+))?'
)
RE_TYPE_ALIAS = re.compile(
    r'^\s*(?:export\s+)?type\s+(\w+)\s*(?:<[^>]+>)?\s*='
)
RE_METHOD = re.compile(
    r'^\s*(?:async\s+)?(\w+)\s*(?:<[^>]+>)?\s*\([^)]*\)'
)

# JS keywords that look like method calls in class bodies
_JS_KEYWORDS = frozenset({
    "if", "for", "while", "switch", "catch", "return",
    "new", "throw", "typeof", "delete", "void", "await",
    "super", "import", "export", "function", "class",
})


class JSExtractor(BaseExtractor):

    def extract_symbols(self, file_path: str, source: str) -> list[SymbolEntry]:
        if not source.strip():
            return []

        lines = source.splitlines()
        symbols: list[SymbolEntry] = []
        brace_depth = 0
        in_class: str | None = None
        class_entry_depth = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            # Interface (TS)
            m = RE_INTERFACE.match(line)
            if m and in_class is None:
                name = m.group(1)
                base = m.group(2)
                sig = f"interface {name} extends {base}" if base else f"interface {name}"
                docstring = self._extract_jsdoc(lines, i)
                symbols.append(SymbolEntry(
                    name=name,
                    kind=SymbolKind.INTERFACE,
                    file_path=file_path,
                    line=i + 1,
                    signature=sig,
                    docstring=self._truncate_docstring(docstring) if docstring else None,
                ))
                brace_depth += line.count("{") - line.count("}")
                continue

            # Type alias (TS)
            m = RE_TYPE_ALIAS.match(line)
            if m and in_class is None:
                name = m.group(1)
                sig = f"type {name}"
                docstring = self._extract_jsdoc(lines, i)
                symbols.append(SymbolEntry(
                    name=name,
                    kind=SymbolKind.CLASS,
                    file_path=file_path,
                    line=i + 1,
                    signature=sig,
                    docstring=self._truncate_docstring(docstring) if docstring else None,
                ))
                brace_depth += line.count("{") - line.count("}")
                continue

            # Class declaration
            m = RE_CLASS.match(line)
            if m and in_class is None:
                name = m.group(1)
                base = m.group(2)
                sig = f"class {name} extends {base}" if base else f"class {name}"
                docstring = self._extract_jsdoc(lines, i)
                symbols.append(SymbolEntry(
                    name=name,
                    kind=SymbolKind.CLASS,
                    file_path=file_path,
                    line=i + 1,
                    signature=sig,
                    docstring=self._truncate_docstring(docstring) if docstring else None,
                ))
                brace_depth += line.count("{") - line.count("}")
                in_class = name
                class_entry_depth = brace_depth
                continue

            # Top-level function
            m = RE_FUNCTION.match(line)
            if m and in_class is None:
                name = m.group(1)
                sig = stripped.rstrip("{").strip()
                # Trim body if present
                if "{" in sig:
                    sig = sig[:sig.index("{")].strip()
                docstring = self._extract_jsdoc(lines, i)
                symbols.append(SymbolEntry(
                    name=name,
                    kind=SymbolKind.FUNCTION,
                    file_path=file_path,
                    line=i + 1,
                    signature=sig,
                    docstring=self._truncate_docstring(docstring) if docstring else None,
                ))
                brace_depth += line.count("{") - line.count("}")
                continue

            # Arrow function
            m = RE_ARROW.match(line)
            if m and in_class is None:
                name = m.group(1)
                sig = stripped.rstrip("{").strip()
                if "{" in sig:
                    sig = sig[:sig.index("{")].strip()
                # Trim trailing semicolons from one-liners
                sig = sig.rstrip(";").strip()
                docstring = self._extract_jsdoc(lines, i)
                symbols.append(SymbolEntry(
                    name=name,
                    kind=SymbolKind.FUNCTION,
                    file_path=file_path,
                    line=i + 1,
                    signature=sig,
                    docstring=self._truncate_docstring(docstring) if docstring else None,
                ))
                brace_depth += line.count("{") - line.count("}")
                continue

            # Inside a class — look for methods
            if in_class is not None:
                pre_depth = brace_depth
                brace_depth += line.count("{") - line.count("}")

                # Check if we've exited the class
                if brace_depth <= class_entry_depth - 1:
                    in_class = None
                    continue

                # Only match methods at class-body depth (not inside method bodies).
                # Use pre-line depth since the method line itself may open a brace.
                if pre_depth != class_entry_depth:
                    continue

                # Skip private methods (# prefix or TS private keyword)
                if stripped.startswith("#") or stripped.startswith("private "):
                    continue

                m = RE_METHOD.match(stripped)
                if m:
                    name = m.group(1)
                    if name in _JS_KEYWORDS:
                        continue
                    docstring = self._extract_jsdoc(lines, i)
                    symbols.append(SymbolEntry(
                        name=name,
                        kind=SymbolKind.METHOD,
                        file_path=file_path,
                        line=i + 1,
                        signature=stripped.rstrip("{").strip(),
                        docstring=self._truncate_docstring(docstring) if docstring else None,
                        parent=in_class,
                    ))
                continue

            # Track brace depth for non-class contexts
            brace_depth += line.count("{") - line.count("}")

        return symbols

    def _extract_jsdoc(self, lines: list[str], decl_line: int) -> str | None:
        """Walk backwards looking for a JSDoc block (/** ... */). Return description only."""
        i = decl_line - 1

        # Skip blank lines
        while i >= 0 and not lines[i].strip():
            i -= 1

        if i < 0:
            return None

        # Check if preceding line ends a JSDoc block
        if not lines[i].strip().endswith("*/"):
            return None

        # Collect the JSDoc block
        doc_lines: list[str] = []
        while i >= 0:
            stripped = lines[i].strip()
            doc_lines.append(stripped)
            if stripped.startswith("/**"):
                break
            i -= 1
        else:
            return None

        doc_lines.reverse()

        # Parse out description lines (skip @param, @returns, etc.)
        description_parts: list[str] = []
        for dl in doc_lines:
            # Strip JSDoc delimiters
            cleaned = dl.strip()
            if cleaned.startswith("/**"):
                cleaned = cleaned[3:]
            if cleaned.endswith("*/"):
                cleaned = cleaned[:-2]
            cleaned = cleaned.strip()
            if cleaned.startswith("*"):
                cleaned = cleaned[1:].strip()

            if not cleaned:
                continue
            if cleaned.startswith("@"):
                continue
            description_parts.append(cleaned)

        if not description_parts:
            return None

        return " ".join(description_parts)
