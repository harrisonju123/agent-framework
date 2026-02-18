"""Tests for semantic search integration in IndexQuery."""

from unittest.mock import MagicMock, patch

import pytest

from agent_framework.indexing.models import (
    CodebaseIndex,
    ModuleEntry,
    SymbolEntry,
    SymbolKind,
)
from agent_framework.indexing.query import IndexQuery
from agent_framework.indexing.store import IndexStore


def _make_index(symbols=None, modules=None) -> CodebaseIndex:
    return CodebaseIndex(
        repo_slug="org/repo",
        commit_sha="abc123",
        language="python",
        total_files=20,
        total_lines=1000,
        symbols=symbols or [],
        modules=modules or [
            ModuleEntry(path="src", description="Source code", language="python", file_count=10),
        ],
        entry_points=["main.py"],
        test_directories=["tests"],
    )


def _make_symbol(name, file_path="src/foo.py", line=1, **kwargs) -> SymbolEntry:
    return SymbolEntry(
        name=name, kind=SymbolKind.FUNCTION,
        file_path=file_path, line=line, **kwargs,
    )


class TestKeywordOnlyFallback:
    """When embedder is None, existing keyword-only behavior is preserved."""

    def test_no_embedder_uses_keyword_only(self, tmp_path):
        store = IndexStore(tmp_path)
        idx = _make_index(symbols=[
            _make_symbol("authenticate", signature="def authenticate(token: str)"),
        ])
        store.save(idx)

        query = IndexQuery(store, embedder=None)
        result = query.query_for_prompt("org/repo", "fix authenticate bug")
        assert "authenticate" in result
        assert "Relevant Symbols" in result

    def test_no_embedder_no_symbols_overview_only(self, tmp_path):
        store = IndexStore(tmp_path)
        idx = _make_index(symbols=[])
        store.save(idx)

        query = IndexQuery(store, embedder=None)
        result = query.query_for_prompt("org/repo", "anything")
        assert "Codebase Overview" in result
        assert "Relevant Symbols" not in result


class TestRRFMergedOutput:
    def test_merged_results_single_section(self, tmp_path):
        """RRF merge produces a single '## Relevant Symbols' section."""
        store = IndexStore(tmp_path)
        idx = _make_index(symbols=[
            _make_symbol("authenticate", file_path="src/auth.py", line=10,
                         signature="def authenticate(token)"),
        ])
        store.save(idx)

        embedder = MagicMock()
        embedder.dimensions = 32
        query = IndexQuery(store, embedder=embedder)

        semantic_results = [
            {"name": "register_user", "kind": "function",
             "file_path": "src/reg.py", "line": 5,
             "signature": "def register_user(data)", "docstring": "", "parent": ""},
        ]
        with patch.object(query, "_try_semantic_query", return_value=semantic_results):
            result = query.query_for_prompt("org/repo", "fix authenticate bug")

        assert result.count("## Relevant Symbols") == 1
        assert "authenticate" in result
        assert "register_user" in result

    def test_semantic_only_when_no_keyword_matches(self, tmp_path):
        """When keywords don't match but semantic does, still shows results."""
        store = IndexStore(tmp_path)
        idx = _make_index(symbols=[
            _make_symbol("unrelated_func"),
        ])
        store.save(idx)

        embedder = MagicMock()
        embedder.dimensions = 32
        query = IndexQuery(store, embedder=embedder)

        semantic_results = [
            {"name": "RegistrationController", "kind": "class",
             "file_path": "src/reg.py", "line": 1,
             "signature": "class RegistrationController", "docstring": "", "parent": ""},
        ]
        # "improve onboarding" has no keyword matches with "unrelated_func"
        with patch.object(query, "_try_semantic_query", return_value=semantic_results):
            result = query.query_for_prompt("org/repo", "improve user onboarding")

        assert "RegistrationController" in result

    def test_empty_semantic_uses_keyword_only(self, tmp_path):
        """When semantic returns empty, fall back to keyword-only."""
        store = IndexStore(tmp_path)
        idx = _make_index(symbols=[
            _make_symbol("authenticate", signature="def authenticate(token)"),
        ])
        store.save(idx)

        embedder = MagicMock()
        embedder.dimensions = 32
        query = IndexQuery(store, embedder=embedder)

        with patch.object(query, "_try_semantic_query", return_value=[]):
            result = query.query_for_prompt("org/repo", "fix authenticate bug")

        assert "authenticate" in result
        assert "Relevant Symbols" in result


class TestBudgetRespected:
    def test_merged_output_within_budget(self, tmp_path):
        store = IndexStore(tmp_path)
        symbols = [
            _make_symbol(f"func_{i}", signature=f"def func_{i}(x)", line=i)
            for i in range(50)
        ]
        idx = _make_index(symbols=symbols)
        store.save(idx)

        embedder = MagicMock()
        embedder.dimensions = 32
        query = IndexQuery(store, embedder=embedder)

        semantic_results = [
            {"name": f"sem_{i}", "kind": "function",
             "file_path": f"src/s{i}.py", "line": i,
             "signature": f"def sem_{i}(x)", "docstring": "", "parent": ""}
            for i in range(20)
        ]
        with patch.object(query, "_try_semantic_query", return_value=semantic_results):
            result = query.query_for_prompt("org/repo", "func related code", max_chars=500)

        assert len(result) <= 500


class TestTrySemanticQuery:
    def test_returns_empty_when_embedder_none(self, tmp_path):
        store = IndexStore(tmp_path)
        query = IndexQuery(store, embedder=None)
        assert query._try_semantic_query("org/repo", "anything") == []

    def test_returns_empty_when_lancedb_missing(self, tmp_path):
        store = IndexStore(tmp_path)
        embedder = MagicMock()
        embedder.dimensions = 32
        query = IndexQuery(store, embedder=embedder)
        # No lancedb dir exists
        assert query._try_semantic_query("org/repo", "anything") == []

    def test_returns_empty_on_exception(self, tmp_path):
        store = IndexStore(tmp_path)
        embedder = MagicMock()
        embedder.dimensions = 32
        query = IndexQuery(store, embedder=embedder)

        with patch("agent_framework.indexing.embeddings.vector_store.VectorStore",
                    side_effect=RuntimeError("boom")):
            assert query._try_semantic_query("org/repo", "anything") == []
