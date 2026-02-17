"""Abstract base class for language-specific symbol extractors."""

import os
import re
from abc import ABC, abstractmethod

from agent_framework.indexing.models import ModuleEntry, SymbolEntry

MAX_FILE_SIZE = 500_000
DOCSTRING_MAX_LEN = 120

KEY_FILE_PATTERNS = [
    "model",
    "handler",
    "service",
    "main",
    "router",
    "controller",
    "config",
    "schema",
]


class BaseExtractor(ABC):

    @abstractmethod
    def extract_symbols(self, file_path: str, source: str) -> list[SymbolEntry]:
        ...

    def extract_module(
        self, dir_path: str, files: list[str], language: str
    ) -> ModuleEntry:
        description = self._find_module_description(dir_path)
        key_files = self._rank_key_files(files)
        return ModuleEntry(
            path=dir_path,
            description=description,
            language=language,
            file_count=len(files),
            key_files=key_files,
        )

    def _find_module_description(self, dir_path: str) -> str:
        # Priority 1: __init__.py module docstring
        init_path = os.path.join(dir_path, "__init__.py")
        try:
            with open(init_path) as f:
                content = f.read()
            match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            if not match:
                match = re.search(r"'''(.*?)'''", content, re.DOTALL)
            if match:
                return match.group(1).strip()
        except OSError:
            pass

        # Priority 2: doc.go package comment
        doc_go_path = os.path.join(dir_path, "doc.go")
        try:
            with open(doc_go_path) as f:
                lines = f.readlines()
            comment_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("//"):
                    comment_lines.append(stripped.lstrip("/").strip())
                elif comment_lines:
                    break
            if comment_lines:
                return " ".join(comment_lines)
        except OSError:
            pass

        # Priority 3: README.md first paragraph
        readme_path = os.path.join(dir_path, "README.md")
        try:
            with open(readme_path) as f:
                content = f.read()
            for paragraph in content.split("\n\n"):
                text = paragraph.strip()
                if text and not text.startswith("#"):
                    return text
        except OSError:
            pass

        return ""

    def _rank_key_files(self, files: list[str]) -> list[str]:
        if not files:
            return []

        def score(filename: str) -> int:
            lower = filename.lower()
            total = 0
            for i, pattern in enumerate(KEY_FILE_PATTERNS):
                if pattern in lower:
                    # Earlier patterns score higher
                    total += len(KEY_FILE_PATTERNS) - i
            return total

        scored = [(f, score(f)) for f in files]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [f for f, s in scored[:5] if s > 0]

    def _truncate_docstring(self, docstring: str) -> str:
        if len(docstring) <= DOCSTRING_MAX_LEN:
            return docstring
        return docstring[: DOCSTRING_MAX_LEN - 3] + "..."
