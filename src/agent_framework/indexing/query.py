"""Query interface for injecting index data into prompts."""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

from agent_framework.indexing.models import CodebaseIndex, SymbolEntry
from agent_framework.indexing.store import IndexStore

_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "and", "but", "or", "if", "while", "because", "until", "that", "this",
    "these", "those", "it", "its", "we", "us", "they", "them", "their",
    "what", "which", "who", "whom",
})

_ACTION_VERBS: frozenset[str] = frozenset({
    "add", "create", "implement", "fix", "update", "remove", "test",
    "review", "move", "document", "refactor", "delete", "change",
    "modify", "write", "build", "deploy", "migrate", "convert",
    "integrate", "configure", "setup", "install", "upgrade", "ensure",
    "make", "use", "set", "get", "run", "check", "verify", "handle",
    "support", "enable", "disable",
})


class IndexQuery:
    """Searches a stored CodebaseIndex and formats results for prompt injection."""

    def __init__(self, store: IndexStore, embedder=None, n_semantic_results: int = 15) -> None:
        self._store = store
        self._embedder = embedder
        self._n_semantic_results = n_semantic_results

    def query_for_prompt(
        self, repo_slug: str, task_goal: str, max_chars: int = 4000
    ) -> str:
        index = self._store.load(repo_slug)
        if index is None:
            return ""

        lines: list[str] = []
        lines.extend(self._format_overview(index))

        keywords = self._extract_keywords(task_goal)
        keyword_scored = []
        if keywords and index.symbols:
            keyword_scored = self._score_symbols(index.symbols, keywords)

        # RRF merge with semantic results when embedder is available
        semantic_results = self._try_semantic_query(repo_slug, task_goal)

        if semantic_results:
            from agent_framework.indexing.embeddings.semantic_query import SemanticQuery
            if keyword_scored:
                merged = SemanticQuery.merge_with_keyword_results(
                    keyword_scored[:30], semantic_results,
                )
            else:
                merged = semantic_results
            if merged:
                overview_text = "\n".join(lines)
                remaining = max_chars - len(overview_text) - 2
                symbol_text = SemanticQuery.format_results(merged, max_chars=remaining)
                if symbol_text:
                    lines.append("")
                    lines.append(symbol_text)
        elif keyword_scored:
            # Keyword-only fallback (original behavior)
            top = keyword_scored[:30]
            if top:
                lines.append("")
                lines.append("## Relevant Symbols")
                for sym, _score in top:
                    lines.append(self._format_symbol(sym))

        return self._truncate("\n".join(lines), max_chars)

    def format_overview_only(self, repo_slug: str) -> str:
        index = self._store.load(repo_slug)
        if index is None:
            return ""
        return "\n".join(self._format_overview(index))

    def _try_semantic_query(self, repo_slug: str, task_goal: str) -> list[dict]:
        """Best-effort semantic search. Returns empty list on any failure."""
        if self._embedder is None:
            return []
        try:
            from agent_framework.indexing.embeddings.vector_store import VectorStore
            from agent_framework.indexing.embeddings.semantic_query import SemanticQuery

            store_path = self._store._slug_dir(repo_slug) / "lancedb"
            if not store_path.exists():
                return []
            vector_store = VectorStore(store_path, dimensions=self._embedder.dimensions)
            sq = SemanticQuery(vector_store, self._embedder)
            return sq.query(task_goal, n_results=self._n_semantic_results)
        except Exception:
            logger.debug("Semantic query failed for %s", repo_slug, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_overview(index: CodebaseIndex) -> list[str]:
        lines = [
            "## Codebase Overview",
            f"- **Repo**: {index.repo_slug}",
            f"- **Language**: {index.language}",
            f"- **Files**: {index.total_files} ({index.total_lines} lines)",
        ]
        if index.entry_points:
            lines.append(f"- **Entry points**: {', '.join(index.entry_points)}")
        if index.test_directories:
            lines.append(f"- **Test dirs**: {', '.join(index.test_directories)}")
        if index.modules:
            lines.append("")
            lines.append("### Modules")
            for mod in index.modules:
                desc = f" — {mod.description}" if mod.description else ""
                lines.append(f"- `{mod.path}/` ({mod.file_count} files){desc}")
                if mod.key_files:
                    for kf in mod.key_files:
                        lines.append(f"  - `{kf}`")
        return lines

    @staticmethod
    def _format_symbol(sym: SymbolEntry) -> str:
        sig = sym.signature or sym.name
        parts = [f"- `{sig}`"]
        if sym.parent:
            parts.append(f"(in {sym.parent})")
        parts.append(f"[{sym.file_path}:{sym.line}]")
        if sym.docstring:
            parts.append(f"-- {sym.docstring}")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Keyword extraction & scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text)
        seen: set[str] = set()
        result: list[str] = []
        for tok in tokens:
            lower = tok.lower()
            if len(lower) < 2:
                continue
            if lower in _STOPWORDS or lower in _ACTION_VERBS:
                continue
            if lower not in seen:
                seen.add(lower)
                result.append(lower)
        return result

    @staticmethod
    def _score_symbols(
        symbols: list[SymbolEntry], keywords: list[str]
    ) -> list[tuple[SymbolEntry, int]]:
        scored: list[tuple[SymbolEntry, int]] = []
        for sym in symbols:
            score = 0
            name_lower = sym.name.lower()
            for kw in keywords:
                if kw in name_lower:
                    score += 3
                # 1 pt for match in other fields
                for field_val in (
                    sym.file_path,
                    sym.signature or "",
                    sym.docstring or "",
                    sym.parent or "",
                ):
                    if kw in field_val.lower():
                        score += 1
            if score > 0:
                scored.append((sym, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Truncation
    # ------------------------------------------------------------------

    _TRUNCATION_SUFFIX = "\n... (truncated to fit prompt budget)"

    @classmethod
    def _truncate(cls, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        # Reserve room for the suffix so total stays within budget
        budget = max_chars - len(cls._TRUNCATION_SUFFIX)
        if budget <= 0:
            return text[:max_chars]
        truncated = text[:budget]
        last_nl = truncated.rfind("\n")
        if last_nl > 0:
            truncated = truncated[:last_nl]
        return truncated + cls._TRUNCATION_SUFFIX
