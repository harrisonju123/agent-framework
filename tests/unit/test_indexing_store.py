"""Tests for IndexStore persistence layer."""

import json

import pytest

from agent_framework.indexing.models import CodebaseIndex, ModuleEntry, SymbolEntry, SymbolKind
from agent_framework.indexing.store import IndexStore


def _make_index(repo_slug="org/repo", commit_sha="abc123") -> CodebaseIndex:
    return CodebaseIndex(
        repo_slug=repo_slug,
        commit_sha=commit_sha,
        language="python",
        total_files=10,
        total_lines=500,
        modules=[
            ModuleEntry(path="src", description="Source", language="python", file_count=5),
        ],
        symbols=[
            SymbolEntry(
                name="Foo",
                kind=SymbolKind.CLASS,
                file_path="src/foo.py",
                line=1,
                signature="class Foo:",
                docstring="A foo class",
            ),
        ],
        entry_points=["main.py"],
        test_directories=["tests"],
    )


class TestSaveLoadRoundtrip:
    def test_all_fields_preserved(self, tmp_path):
        store = IndexStore(tmp_path)
        idx = _make_index()
        store.save(idx)
        loaded = store.load("org/repo")

        assert loaded is not None
        assert loaded.repo_slug == idx.repo_slug
        assert loaded.commit_sha == idx.commit_sha
        assert loaded.language == idx.language
        assert loaded.total_files == idx.total_files
        assert loaded.total_lines == idx.total_lines
        assert len(loaded.modules) == 1
        assert loaded.modules[0].path == "src"
        assert len(loaded.symbols) == 1
        assert loaded.symbols[0].name == "Foo"
        assert loaded.entry_points == ["main.py"]
        assert loaded.test_directories == ["tests"]

    def test_multiple_saves_overwrite(self, tmp_path):
        store = IndexStore(tmp_path)
        store.save(_make_index(commit_sha="v1"))
        store.save(_make_index(commit_sha="v2"))
        loaded = store.load("org/repo")
        assert loaded.commit_sha == "v2"


class TestMissingIndex:
    def test_load_missing_returns_none(self, tmp_path):
        store = IndexStore(tmp_path)
        assert store.load("nonexistent/repo") is None


class TestIsStale:
    def test_stale_when_missing(self, tmp_path):
        store = IndexStore(tmp_path)
        assert store.is_stale("org/repo", "sha1") is True

    def test_stale_when_sha_differs(self, tmp_path):
        store = IndexStore(tmp_path)
        store.save(_make_index(commit_sha="old"))
        assert store.is_stale("org/repo", "new") is True

    def test_not_stale_when_sha_matches(self, tmp_path):
        store = IndexStore(tmp_path)
        store.save(_make_index(commit_sha="same"))
        assert store.is_stale("org/repo", "same") is False


class TestCorruptIndex:
    def test_corrupt_json_returns_none(self, tmp_path):
        store = IndexStore(tmp_path)
        # Write a valid index first to create directory structure
        store.save(_make_index())
        # Overwrite with garbage
        idx_path = store._index_path("org/repo")
        idx_path.write_text("not valid json {{{")
        assert store.load("org/repo") is None

    def test_invalid_schema_returns_none(self, tmp_path):
        store = IndexStore(tmp_path)
        store.save(_make_index())
        idx_path = store._index_path("org/repo")
        # Valid JSON but wrong schema
        idx_path.write_text(json.dumps({"wrong": "schema"}))
        assert store.load("org/repo") is None


class TestSlugSanitization:
    def test_slash_replaced_with_double_underscore(self, tmp_path):
        store = IndexStore(tmp_path)
        idx = _make_index(repo_slug="myorg/myrepo")
        store.save(idx)
        expected_dir = tmp_path / ".agent-communication" / "indexes" / "myorg__myrepo"
        assert expected_dir.exists()
        assert (expected_dir / "codebase_index.json").exists()


class TestParentDirCreation:
    def test_dirs_created_on_first_save(self, tmp_path):
        store = IndexStore(tmp_path)
        idx = _make_index(repo_slug="brand/new")
        store.save(idx)
        loaded = store.load("brand/new")
        assert loaded is not None
        assert loaded.repo_slug == "brand/new"
