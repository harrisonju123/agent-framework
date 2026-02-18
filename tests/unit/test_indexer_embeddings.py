"""Tests for embedding integration in CodebaseIndexer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_framework.indexing.indexer import CodebaseIndexer
from agent_framework.indexing.models import CodebaseIndex, SymbolEntry, SymbolKind
from agent_framework.indexing.store import IndexStore


def _make_index(repo_slug="org/repo", commit_sha="abc123", **kwargs) -> CodebaseIndex:
    return CodebaseIndex(
        repo_slug=repo_slug,
        commit_sha=commit_sha,
        language="python",
        total_files=5,
        total_lines=100,
        symbols=kwargs.get("symbols", []),
        modules=kwargs.get("modules", []),
    )


class TestTryEmbedIndex:
    def test_noop_when_no_embedder(self, tmp_path):
        store = IndexStore(tmp_path)
        indexer = CodebaseIndexer(store=store, embedder=None)
        index = _make_index()
        # Should not raise
        indexer._try_embed_index(index)

    def test_calls_rebuild_when_changed_files_none(self, tmp_path):
        store = IndexStore(tmp_path)
        embedder = MagicMock()
        embedder.dimensions = 32
        indexer = CodebaseIndexer(store=store, embedder=embedder)
        index = _make_index()
        store.save(index)

        with patch("agent_framework.indexing.embeddings.vector_store.VectorStore") as MockVS:
            mock_vs = MagicMock()
            MockVS.return_value = mock_vs
            mock_vs.is_stale.return_value = True

            indexer._try_embed_index(index, changed_files=None)

            mock_vs.rebuild.assert_called_once_with(index, embedder)
            mock_vs.update_incremental.assert_not_called()

    def test_calls_incremental_when_changed_files_provided(self, tmp_path):
        store = IndexStore(tmp_path)
        embedder = MagicMock()
        embedder.dimensions = 32
        indexer = CodebaseIndexer(store=store, embedder=embedder)
        index = _make_index()
        store.save(index)

        with patch("agent_framework.indexing.embeddings.vector_store.VectorStore") as MockVS:
            mock_vs = MagicMock()
            MockVS.return_value = mock_vs
            mock_vs.is_stale.return_value = False

            changed = {"src/a.py", "src/b.py"}
            deleted = {"src/c.py"}
            indexer._try_embed_index(
                index, changed_files=changed, deleted_files=deleted,
                prior_sha="old_sha",
            )

            mock_vs.update_incremental.assert_called_once_with(
                index, embedder, changed, deleted,
            )
            mock_vs.rebuild.assert_not_called()

    def test_falls_back_to_rebuild_when_stale(self, tmp_path):
        """Even with changed_files, if vector store is stale vs prior SHA, do full rebuild."""
        store = IndexStore(tmp_path)
        embedder = MagicMock()
        embedder.dimensions = 32
        indexer = CodebaseIndexer(store=store, embedder=embedder)
        index = _make_index()
        store.save(index)

        with patch("agent_framework.indexing.embeddings.vector_store.VectorStore") as MockVS:
            mock_vs = MagicMock()
            MockVS.return_value = mock_vs
            mock_vs.is_stale.return_value = True

            indexer._try_embed_index(
                index, changed_files={"src/a.py"}, prior_sha="old_sha",
            )

            mock_vs.rebuild.assert_called_once()

    def test_falls_back_to_rebuild_when_no_prior_sha(self, tmp_path):
        """changed_files without prior_sha means we can't verify vector store state."""
        store = IndexStore(tmp_path)
        embedder = MagicMock()
        embedder.dimensions = 32
        indexer = CodebaseIndexer(store=store, embedder=embedder)
        index = _make_index()
        store.save(index)

        with patch("agent_framework.indexing.embeddings.vector_store.VectorStore") as MockVS:
            mock_vs = MagicMock()
            MockVS.return_value = mock_vs

            indexer._try_embed_index(
                index, changed_files={"src/a.py"}, prior_sha=None,
            )

            mock_vs.rebuild.assert_called_once()
            mock_vs.is_stale.assert_not_called()

    def test_swallows_exceptions(self, tmp_path):
        store = IndexStore(tmp_path)
        embedder = MagicMock()
        embedder.dimensions = 32
        indexer = CodebaseIndexer(store=store, embedder=embedder)
        index = _make_index()
        store.save(index)

        with patch("agent_framework.indexing.embeddings.vector_store.VectorStore",
                    side_effect=RuntimeError("lancedb import error")):
            # Should not raise
            indexer._try_embed_index(index)


class TestDiffFiles:
    def test_parses_changed_and_deleted(self, tmp_path):
        with patch("agent_framework.indexing.indexer.run_git_command") as mock_git:
            mock_git.return_value = MagicMock(
                stdout="M\tsrc/a.py\nD\tsrc/b.py\nA\tsrc/c.py\n"
            )
            changed, deleted = CodebaseIndexer._diff_files(tmp_path, "sha1", "sha2")

            assert changed == {"src/a.py", "src/c.py"}
            assert deleted == {"src/b.py"}

    def test_returns_none_on_failure(self, tmp_path):
        with patch("agent_framework.indexing.indexer.run_git_command",
                    side_effect=RuntimeError("git error")):
            changed, deleted = CodebaseIndexer._diff_files(tmp_path, "sha1", "sha2")
            assert changed is None
            assert deleted is None

    def test_empty_diff(self, tmp_path):
        with patch("agent_framework.indexing.indexer.run_git_command") as mock_git:
            mock_git.return_value = MagicMock(stdout="")
            changed, deleted = CodebaseIndexer._diff_files(tmp_path, "sha1", "sha2")
            assert changed == set()
            assert deleted == set()

    def test_renames_tracked_as_delete_plus_change(self, tmp_path):
        with patch("agent_framework.indexing.indexer.run_git_command") as mock_git:
            mock_git.return_value = MagicMock(
                stdout="R100\tsrc/old.py\tsrc/new.py\nM\tsrc/other.py\n"
            )
            changed, deleted = CodebaseIndexer._diff_files(tmp_path, "sha1", "sha2")
            assert changed == {"src/new.py", "src/other.py"}
            assert deleted == {"src/old.py"}


class TestEnsureIndexedWithEmbeddings:
    def test_embed_called_after_save_with_no_prior(self, tmp_path):
        """First index: _try_embed_index called with prior_sha=None."""
        store = IndexStore(tmp_path)
        embedder = MagicMock()
        embedder.dimensions = 32
        indexer = CodebaseIndexer(store=store, embedder=embedder)

        with patch.object(indexer, "_build_index") as mock_build, \
             patch.object(indexer, "_try_embed_index") as mock_embed, \
             patch("agent_framework.indexing.indexer.run_git_command") as mock_git:
            mock_git.return_value = MagicMock(stdout="newsha\n")
            index = _make_index(commit_sha="newsha")
            mock_build.return_value = index

            result = indexer._ensure_indexed_inner("org/repo", tmp_path)

            assert result is not None
            mock_embed.assert_called_once()
            call_kwargs = mock_embed.call_args
            assert call_kwargs[1]["prior_sha"] is None

    def test_embed_called_with_prior_sha_on_update(self, tmp_path):
        """Updated index: _try_embed_index called with prior_sha set to old SHA."""
        store = IndexStore(tmp_path)
        old_index = _make_index(commit_sha="oldsha")
        store.save(old_index)

        embedder = MagicMock()
        embedder.dimensions = 32
        indexer = CodebaseIndexer(store=store, embedder=embedder)

        with patch.object(indexer, "_build_index") as mock_build, \
             patch.object(indexer, "_try_embed_index") as mock_embed, \
             patch("agent_framework.indexing.indexer.run_git_command") as mock_git:
            mock_git.return_value = MagicMock(stdout="newsha\n")
            new_index = _make_index(commit_sha="newsha")
            mock_build.return_value = new_index

            result = indexer._ensure_indexed_inner("org/repo", tmp_path)

            assert result is not None
            mock_embed.assert_called_once()
            call_kwargs = mock_embed.call_args
            assert call_kwargs[1]["prior_sha"] == "oldsha"

    def test_embed_not_called_when_cached(self, tmp_path):
        """When index is cached (SHA matches), no embedding happens."""
        store = IndexStore(tmp_path)
        index = _make_index(commit_sha="sha123")
        store.save(index)

        embedder = MagicMock()
        embedder.dimensions = 32
        indexer = CodebaseIndexer(store=store, embedder=embedder)

        with patch.object(indexer, "_try_embed_index") as mock_embed, \
             patch("agent_framework.indexing.indexer.run_git_command") as mock_git:
            mock_git.return_value = MagicMock(stdout="sha123\n")

            result = indexer._ensure_indexed_inner("org/repo", tmp_path)

            assert result is not None
            mock_embed.assert_not_called()
