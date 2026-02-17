"""Tests for IndexQuery keyword extraction, scoring, and formatting."""

import pytest

from agent_framework.indexing.models import (
    CodebaseIndex,
    ModuleEntry,
    SymbolEntry,
    SymbolKind,
)
from agent_framework.indexing.query import IndexQuery
from agent_framework.indexing.store import IndexStore


def _make_index(
    symbols: list[SymbolEntry] | None = None,
    modules: list[ModuleEntry] | None = None,
) -> CodebaseIndex:
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


def _make_symbol(
    name: str,
    kind: SymbolKind = SymbolKind.FUNCTION,
    file_path: str = "src/foo.py",
    line: int = 1,
    signature: str | None = None,
    docstring: str | None = None,
    parent: str | None = None,
) -> SymbolEntry:
    return SymbolEntry(
        name=name,
        kind=kind,
        file_path=file_path,
        line=line,
        signature=signature,
        docstring=docstring,
        parent=parent,
    )


class TestKeywordExtraction:
    def test_stopword_removal(self):
        kws = IndexQuery._extract_keywords("the quick brown fox is very fast")
        assert "the" not in kws
        assert "is" not in kws
        assert "very" not in kws
        assert "quick" in kws
        assert "brown" in kws

    def test_action_verb_removal(self):
        kws = IndexQuery._extract_keywords("add authentication to the login handler")
        assert "add" not in kws
        assert "authentication" in kws
        assert "login" in kws
        assert "handler" in kws

    def test_dedup(self):
        kws = IndexQuery._extract_keywords("user User USER")
        assert kws.count("user") == 1

    def test_min_length(self):
        kws = IndexQuery._extract_keywords("a b ab cd")
        assert "a" not in kws
        assert "b" not in kws
        assert "ab" in kws
        assert "cd" in kws

    def test_all_stopwords_returns_empty(self):
        kws = IndexQuery._extract_keywords("the a an is are")
        assert kws == []


class TestSymbolScoring:
    def test_name_match_scores_3(self):
        sym = _make_symbol("auth_handler", file_path="src/unrelated.py")
        scored = IndexQuery._score_symbols([sym], ["auth"])
        assert len(scored) == 1
        # 3 for name match only (file_path doesn't match)
        assert scored[0][1] == 3

    def test_other_field_match_scores_1(self):
        sym = _make_symbol(
            "something",
            file_path="src/auth.py",
            docstring="Handles authentication",
        )
        scored = IndexQuery._score_symbols([sym], ["auth"])
        assert len(scored) == 1
        # 1 for file_path + 1 for docstring
        assert scored[0][1] == 2

    def test_no_match_excluded(self):
        sym = _make_symbol("unrelated")
        scored = IndexQuery._score_symbols([sym], ["auth"])
        assert len(scored) == 0

    def test_top_30_cap(self):
        symbols = [_make_symbol(f"auth_{i}") for i in range(50)]
        scored = IndexQuery._score_symbols(symbols, ["auth"])
        assert len(scored) == 50  # scoring returns all matches
        # query_for_prompt takes top 30 — tested via integration


class TestOverviewFormatting:
    def test_overview_contains_repo_info(self, tmp_path):
        store = IndexStore(tmp_path)
        idx = _make_index()
        store.save(idx)
        query = IndexQuery(store)

        result = query.format_overview_only("org/repo")
        assert "org/repo" in result
        assert "python" in result
        assert "20" in result  # total_files
        assert "1000" in result  # total_lines
        assert "main.py" in result
        assert "tests" in result
        assert "src/" in result

    def test_overview_with_module_description(self, tmp_path):
        store = IndexStore(tmp_path)
        idx = _make_index(modules=[
            ModuleEntry(path="core", description="Core logic", language="python", file_count=5),
        ])
        store.save(idx)
        query = IndexQuery(store)

        result = query.format_overview_only("org/repo")
        assert "Core logic" in result


class TestFormatOverviewOnly:
    def test_no_index_returns_empty(self, tmp_path):
        store = IndexStore(tmp_path)
        query = IndexQuery(store)
        assert query.format_overview_only("missing/repo") == ""


class TestQueryForPrompt:
    def test_no_index_returns_empty(self, tmp_path):
        store = IndexStore(tmp_path)
        query = IndexQuery(store)
        assert query.query_for_prompt("missing/repo", "anything") == ""

    def test_includes_matching_symbols(self, tmp_path):
        store = IndexStore(tmp_path)
        idx = _make_index(symbols=[
            _make_symbol("authenticate_user", signature="def authenticate_user(token: str) -> bool"),
            _make_symbol("process_payment"),
        ])
        store.save(idx)
        query = IndexQuery(store)

        result = query.query_for_prompt("org/repo", "fix authenticate_user bug")
        assert "authenticate_user" in result
        assert "Relevant Symbols" in result

    def test_all_stopword_goal_gives_overview_only(self, tmp_path):
        store = IndexStore(tmp_path)
        idx = _make_index(symbols=[
            _make_symbol("something"),
        ])
        store.save(idx)
        query = IndexQuery(store)

        result = query.query_for_prompt("org/repo", "the a an is are")
        assert "Codebase Overview" in result
        # No symbol section since no keywords match
        assert "Relevant Symbols" not in result

    def test_top_30_symbols(self, tmp_path):
        store = IndexStore(tmp_path)
        symbols = [_make_symbol(f"auth_{i}") for i in range(50)]
        idx = _make_index(symbols=symbols)
        store.save(idx)
        query = IndexQuery(store)

        result = query.query_for_prompt("org/repo", "auth related code", max_chars=50000)
        # Should cap at 30 symbols
        assert result.count("auth_") <= 30


class TestTruncation:
    def test_truncates_at_line_boundary(self, tmp_path):
        store = IndexStore(tmp_path)
        symbols = [
            _make_symbol(f"func_{i}", signature=f"def func_{i}(x): pass", docstring=f"Does thing {i}")
            for i in range(100)
        ]
        idx = _make_index(symbols=symbols)
        store.save(idx)
        query = IndexQuery(store)

        result = query.query_for_prompt("org/repo", "func related", max_chars=500)
        assert len(result) <= 500
        assert "truncated to fit prompt budget" in result

    def test_no_truncation_when_under_limit(self, tmp_path):
        store = IndexStore(tmp_path)
        idx = _make_index(symbols=[_make_symbol("foo")])
        store.save(idx)
        query = IndexQuery(store)

        result = query.query_for_prompt("org/repo", "foo", max_chars=50000)
        assert "truncated" not in result
