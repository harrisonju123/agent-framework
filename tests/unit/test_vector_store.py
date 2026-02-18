"""Tests for VectorStore — rebuild, query, is_stale, incremental update."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

lancedb = pytest.importorskip("lancedb", reason="lancedb not installed")

from agent_framework.indexing.embeddings.vector_store import (
    VectorStore,
    _symbol_doc_text,
    _module_doc_text,
)
from agent_framework.indexing.models import (
    CodebaseIndex,
    ModuleEntry,
    SymbolEntry,
    SymbolKind,
)


def _make_index(
    symbols=None,
    modules=None,
    commit_sha="abc123",
) -> CodebaseIndex:
    return CodebaseIndex(
        repo_slug="org/repo",
        commit_sha=commit_sha,
        language="python",
        total_files=10,
        total_lines=500,
        symbols=symbols or [],
        modules=modules or [],
    )


def _make_symbol(name="foo", file_path="src/foo.py", line=1, **kwargs) -> SymbolEntry:
    return SymbolEntry(
        name=name, kind=SymbolKind.FUNCTION, file_path=file_path,
        line=line, **kwargs,
    )


def _make_module(path="src", **kwargs) -> ModuleEntry:
    return ModuleEntry(path=path, language="python", file_count=5, **kwargs)


def _mock_embedder(dimensions=256):
    """Return a mock embedder that produces random vectors."""
    embedder = MagicMock()

    def embed_texts(texts):
        vecs = np.random.randn(len(texts), dimensions).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        return vecs.tolist()

    embedder.embed_texts = MagicMock(side_effect=embed_texts)
    return embedder


class TestDocTextBuilders:
    def test_symbol_doc_text(self):
        sym = _make_symbol(
            name="authenticate",
            file_path="src/auth.py",
            signature="def authenticate(token: str) -> bool",
            docstring="Validates OAuth tokens",
        )
        text = _symbol_doc_text(sym)
        assert "function authenticate in src/auth.py" in text
        assert "def authenticate(token: str) -> bool" in text
        assert "Validates OAuth tokens" in text

    def test_module_doc_text(self):
        mod = _make_module(
            path="src/auth",
            description="Authentication module",
            key_files=["auth.py", "oauth.py"],
        )
        text = _module_doc_text(mod)
        assert "module src/auth/" in text
        assert "5 files" in text
        assert "Authentication module" in text
        assert "auth.py" in text


class TestVectorStoreRebuild:
    def test_rebuild_creates_table(self, tmp_path):
        store = VectorStore(tmp_path / "lancedb", dimensions=32)
        embedder = _mock_embedder(dimensions=32)
        index = _make_index(
            symbols=[_make_symbol("foo"), _make_symbol("bar", file_path="src/bar.py", line=10)],
            modules=[_make_module()],
        )
        store.rebuild(index, embedder)
        assert store._has_table()
        embedder.embed_texts.assert_called_once()

    def test_rebuild_replaces_existing(self, tmp_path):
        store = VectorStore(tmp_path / "lancedb", dimensions=32)
        embedder = _mock_embedder(dimensions=32)
        index1 = _make_index(symbols=[_make_symbol("foo")])
        store.rebuild(index1, embedder)

        index2 = _make_index(
            symbols=[_make_symbol("bar"), _make_symbol("baz")],
            commit_sha="def456",
        )
        store.rebuild(index2, embedder)
        assert not store.is_stale("def456")

    def test_rebuild_empty_index_no_table(self, tmp_path):
        store = VectorStore(tmp_path / "lancedb", dimensions=32)
        embedder = _mock_embedder(dimensions=32)
        index = _make_index()
        store.rebuild(index, embedder)
        assert not store._has_table()

    def test_rebuild_handles_embed_failure(self, tmp_path):
        store = VectorStore(tmp_path / "lancedb", dimensions=32)
        embedder = MagicMock()
        embedder.embed_texts.return_value = None
        index = _make_index(symbols=[_make_symbol("foo")])
        store.rebuild(index, embedder)
        assert not store._has_table()


class TestVectorStoreQuery:
    def test_query_returns_results(self, tmp_path):
        store = VectorStore(tmp_path / "lancedb", dimensions=32)
        embedder = _mock_embedder(dimensions=32)
        index = _make_index(
            symbols=[
                _make_symbol("authenticate", file_path="src/auth.py", line=10,
                             signature="def authenticate(token)"),
                _make_symbol("process_payment", file_path="src/pay.py", line=20),
            ],
        )
        store.rebuild(index, embedder)

        query_vec = np.random.randn(32).tolist()
        results = store.query(query_vec, n_results=5)
        assert len(results) == 2
        assert all("file_path" in r for r in results)
        assert all("name" in r for r in results)

    def test_query_empty_store_returns_empty(self, tmp_path):
        store = VectorStore(tmp_path / "lancedb", dimensions=32)
        query_vec = np.random.randn(32).tolist()
        results = store.query(query_vec)
        assert results == []

    def test_query_respects_n_results(self, tmp_path):
        store = VectorStore(tmp_path / "lancedb", dimensions=32)
        embedder = _mock_embedder(dimensions=32)
        symbols = [_make_symbol(f"func_{i}", line=i) for i in range(20)]
        index = _make_index(symbols=symbols)
        store.rebuild(index, embedder)

        query_vec = np.random.randn(32).tolist()
        results = store.query(query_vec, n_results=5)
        assert len(results) == 5


class TestIsStale:
    def test_stale_when_no_table(self, tmp_path):
        store = VectorStore(tmp_path / "lancedb", dimensions=32)
        assert store.is_stale("any_sha") is True

    def test_not_stale_when_sha_matches(self, tmp_path):
        store = VectorStore(tmp_path / "lancedb", dimensions=32)
        embedder = _mock_embedder(dimensions=32)
        index = _make_index(symbols=[_make_symbol()], commit_sha="sha123")
        store.rebuild(index, embedder)
        assert store.is_stale("sha123") is False

    def test_stale_when_sha_differs(self, tmp_path):
        store = VectorStore(tmp_path / "lancedb", dimensions=32)
        embedder = _mock_embedder(dimensions=32)
        index = _make_index(symbols=[_make_symbol()], commit_sha="sha123")
        store.rebuild(index, embedder)
        assert store.is_stale("sha456") is True


class TestIncrementalUpdate:
    def test_incremental_adds_new_symbols(self, tmp_path):
        store = VectorStore(tmp_path / "lancedb", dimensions=32)
        embedder = _mock_embedder(dimensions=32)

        index1 = _make_index(
            symbols=[_make_symbol("foo", file_path="src/a.py", line=1)],
            commit_sha="sha1",
        )
        store.rebuild(index1, embedder)

        index2 = _make_index(
            symbols=[
                _make_symbol("foo", file_path="src/a.py", line=1),
                _make_symbol("bar", file_path="src/b.py", line=1),
            ],
            commit_sha="sha2",
        )
        store.update_incremental(
            index2, embedder,
            changed_files={"src/b.py"},
            deleted_files=set(),
        )

        query_vec = np.random.randn(32).tolist()
        results = store.query(query_vec, n_results=10)
        names = {r["name"] for r in results}
        assert "foo" in names
        assert "bar" in names

    def test_incremental_removes_deleted_files(self, tmp_path):
        store = VectorStore(tmp_path / "lancedb", dimensions=32)
        embedder = _mock_embedder(dimensions=32)

        index1 = _make_index(
            symbols=[
                _make_symbol("foo", file_path="src/a.py", line=1),
                _make_symbol("bar", file_path="src/b.py", line=1),
            ],
            commit_sha="sha1",
        )
        store.rebuild(index1, embedder)

        index2 = _make_index(
            symbols=[_make_symbol("foo", file_path="src/a.py", line=1)],
            commit_sha="sha2",
        )
        store.update_incremental(
            index2, embedder,
            changed_files=set(),
            deleted_files={"src/b.py"},
        )

        query_vec = np.random.randn(32).tolist()
        results = store.query(query_vec, n_results=10)
        names = {r["name"] for r in results}
        assert "foo" in names
        assert "bar" not in names

    def test_incremental_falls_back_to_rebuild_when_no_table(self, tmp_path):
        store = VectorStore(tmp_path / "lancedb", dimensions=32)
        embedder = _mock_embedder(dimensions=32)
        index = _make_index(
            symbols=[_make_symbol("foo")],
            commit_sha="sha1",
        )
        store.update_incremental(index, embedder, changed_files={"src/foo.py"}, deleted_files=set())
        assert store._has_table()
