"""Tests for SemanticQuery — query delegation, RRF merge, format_results."""

from unittest.mock import MagicMock

import pytest

from agent_framework.indexing.embeddings.semantic_query import SemanticQuery
from agent_framework.indexing.models import SymbolEntry, SymbolKind


def _sym(name, file_path="src/foo.py", line=1, **kwargs) -> SymbolEntry:
    return SymbolEntry(
        name=name, kind=SymbolKind.FUNCTION,
        file_path=file_path, line=line, **kwargs,
    )


def _sem_result(name, file_path="src/foo.py", line=1, **kwargs) -> dict:
    return {
        "name": name, "kind": "function",
        "file_path": file_path, "line": line,
        "signature": kwargs.get("signature", ""),
        "docstring": kwargs.get("docstring", ""),
        "parent": kwargs.get("parent", ""),
    }


class TestQuery:
    def test_delegates_to_vector_store(self):
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.1, 0.2, 0.3]
        vs = MagicMock()
        vs.query.return_value = [_sem_result("foo")]

        sq = SemanticQuery(vs, embedder)
        results = sq.query("improve onboarding", n_results=10)

        embedder.embed_query.assert_called_once_with("improve onboarding")
        vs.query.assert_called_once_with([0.1, 0.2, 0.3], n_results=10)
        assert len(results) == 1
        assert results[0]["name"] == "foo"

    def test_returns_empty_when_embedding_fails(self):
        embedder = MagicMock()
        embedder.embed_query.return_value = None
        vs = MagicMock()

        sq = SemanticQuery(vs, embedder)
        results = sq.query("test")
        assert results == []
        vs.query.assert_not_called()


class TestRRFMerge:
    def test_keyword_only_items(self):
        """Items only in keyword results should appear."""
        keyword = [(_sym("auth", file_path="src/auth.py", line=10), 5)]
        semantic = []
        merged = SemanticQuery.merge_with_keyword_results(keyword, semantic)
        assert len(merged) == 1
        assert merged[0]["name"] == "auth"

    def test_semantic_only_items(self):
        """Items only in semantic results should appear."""
        keyword = []
        semantic = [_sem_result("register", file_path="src/reg.py", line=20)]
        merged = SemanticQuery.merge_with_keyword_results(keyword, semantic)
        assert len(merged) == 1
        assert merged[0]["name"] == "register"

    def test_overlapping_items_get_boosted(self):
        """Items in both rankers get higher RRF score than items in only one."""
        # "auth" appears in both, "login" only in keyword, "register" only in semantic
        keyword = [
            (_sym("auth", file_path="src/auth.py", line=10), 5),
            (_sym("login", file_path="src/login.py", line=1), 3),
        ]
        semantic = [
            _sem_result("auth", file_path="src/auth.py", line=10),
            _sem_result("register", file_path="src/reg.py", line=1),
        ]
        merged = SemanticQuery.merge_with_keyword_results(keyword, semantic)

        names = [m["name"] for m in merged]
        # "auth" should be first since it appears in both rankers
        assert names[0] == "auth"
        assert len(merged) == 3

    def test_rrf_score_calculation(self):
        """Verify RRF formula: score = sum(1/(rank+k)) with k=60."""
        keyword = [
            (_sym("a", file_path="a.py", line=1), 10),  # rank 0
            (_sym("b", file_path="b.py", line=1), 5),   # rank 1
        ]
        semantic = [
            _sem_result("b", file_path="b.py", line=1),  # rank 0
            _sem_result("c", file_path="c.py", line=1),  # rank 1
        ]

        merged = SemanticQuery.merge_with_keyword_results(keyword, semantic, k=60)

        # "b" appears at rank 1 in keyword (1/61) and rank 0 in semantic (1/60)
        # = 1/61 + 1/60 ≈ 0.0331
        # "a" appears at rank 0 in keyword only (1/60) ≈ 0.0167
        # "c" appears at rank 1 in semantic only (1/61) ≈ 0.0164
        names = [m["name"] for m in merged]
        assert names == ["b", "a", "c"]

    def test_empty_both_returns_empty(self):
        merged = SemanticQuery.merge_with_keyword_results([], [])
        assert merged == []


class TestFormatResults:
    def test_basic_formatting(self):
        results = [
            _sem_result("authenticate", file_path="src/auth.py", line=10,
                        signature="def authenticate(token: str) -> bool",
                        docstring="Validates tokens"),
        ]
        text = SemanticQuery.format_results(results, max_chars=5000)
        assert "## Relevant Symbols" in text
        assert "def authenticate(token: str) -> bool" in text
        assert "src/auth.py:10" in text
        assert "Validates tokens" in text

    def test_respects_budget(self):
        results = [
            _sem_result(f"func_{i}", file_path=f"src/f{i}.py", line=i,
                        signature=f"def func_{i}(x: int) -> str")
            for i in range(100)
        ]
        text = SemanticQuery.format_results(results, max_chars=300)
        assert len(text) <= 300

    def test_empty_results_returns_empty(self):
        assert SemanticQuery.format_results([], max_chars=5000) == ""

    def test_parent_included(self):
        results = [
            _sem_result("method", file_path="src/a.py", line=1,
                        signature="def method(self)", parent="MyClass"),
        ]
        text = SemanticQuery.format_results(results, max_chars=5000)
        assert "(in MyClass)" in text
