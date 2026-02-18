"""Composes Embedder + VectorStore with Reciprocal Rank Fusion merge."""

import logging
from collections import defaultdict

from agent_framework.indexing.embeddings.embedder import Embedder
from agent_framework.indexing.embeddings.vector_store import VectorStore
from agent_framework.indexing.models import SymbolEntry

logger = logging.getLogger(__name__)


class SemanticQuery:
    """Embedding-based semantic search with RRF merge support."""

    def __init__(self, vector_store: VectorStore, embedder: Embedder) -> None:
        self._vector_store = vector_store
        self._embedder = embedder

    def query(self, goal: str, n_results: int = 15) -> list[dict]:
        """Embed goal and query vector store."""
        embedding = self._embedder.embed_query(goal)
        if embedding is None:
            return []
        return self._vector_store.query(embedding, n_results=n_results)

    @staticmethod
    def merge_with_keyword_results(
        keyword_ranked: list[tuple[SymbolEntry, int]],
        semantic_ranked: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion over keyword and semantic results.

        RRF: score = sum(1 / (rank_i + k)) across all rankers.
        k=60 is the standard default that dampens high-rank dominance.
        """
        scores: dict[tuple[str, int], float] = defaultdict(float)
        # Track metadata for each key so we can return rich results
        items: dict[tuple[str, int], dict] = {}

        for rank, (sym, kw_score) in enumerate(keyword_ranked):
            key = (sym.file_path, sym.line)
            scores[key] += 1.0 / (rank + k)
            if key not in items:
                items[key] = {
                    "name": sym.name,
                    "kind": str(sym.kind),
                    "file_path": sym.file_path,
                    "line": sym.line,
                    "signature": sym.signature or "",
                    "docstring": sym.docstring or "",
                    "parent": sym.parent or "",
                }

        for rank, item in enumerate(semantic_ranked):
            key = (item["file_path"], item["line"])
            scores[key] += 1.0 / (rank + k)
            if key not in items:
                items[key] = item

        # Sort by fused score descending
        ranked_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
        return [items[key] for key in ranked_keys]

    @staticmethod
    def format_results(results: list[dict], max_chars: int) -> str:
        """Format merged results for prompt injection."""
        if not results:
            return ""

        lines = ["## Relevant Symbols"]
        budget = max_chars - len(lines[0]) - 1  # newline

        for item in results:
            sig = item.get("signature") or item.get("name", "")
            parts = [f"- `{sig}`"]
            parent = item.get("parent", "")
            if parent:
                parts.append(f"(in {parent})")
            parts.append(f"[{item['file_path']}:{item['line']}]")
            doc = item.get("docstring", "")
            if doc:
                parts.append(f"-- {doc}")
            line = " ".join(parts)

            if budget - len(line) - 1 < 0:
                break
            lines.append(line)
            budget -= len(line) + 1  # +1 for newline

        return "\n".join(lines)
