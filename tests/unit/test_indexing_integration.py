"""Integration tests — git repo → indexer → query round-trip."""

import subprocess

import pytest

from agent_framework.indexing.indexer import CodebaseIndexer
from agent_framework.indexing.query import IndexQuery
from agent_framework.indexing.store import IndexStore


def _init_git_repo(path):
    """Create a git repo with initial commit."""
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=path, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=path, capture_output=True, check=True,
    )


def _commit_all(path, msg="commit"):
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", msg, "--allow-empty"],
        cwd=path, capture_output=True, check=True,
    )


@pytest.fixture
def git_repo(tmp_path):
    """Temp git repo with a few Python files."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    # src/models.py
    (repo / "src").mkdir()
    (repo / "src" / "__init__.py").write_text('"""Source package."""\n')
    (repo / "src" / "models.py").write_text(
        'class User:\n    """A user in the system."""\n    pass\n\n'
        'class Account:\n    """A billing account."""\n    pass\n'
    )

    # src/handler.py
    (repo / "src" / "handler.py").write_text(
        'def handle_request(req):\n    """Process incoming request."""\n    return req\n\n'
        'def validate_input(data):\n    """Validate user input."""\n    return True\n'
    )

    # main.py
    (repo / "main.py").write_text(
        'def main():\n    print("hello")\n\nif __name__ == "__main__":\n    main()\n'
    )

    # tests/test_models.py
    (repo / "tests").mkdir()
    (repo / "tests" / "__init__.py").write_text("")
    (repo / "tests" / "test_models.py").write_text(
        "def test_user():\n    pass\n"
    )

    _commit_all(repo, "initial commit")
    return repo


@pytest.fixture
def store(tmp_path):
    return IndexStore(tmp_path / "store")


class TestEnsureIndexed:
    def test_indexes_python_repo(self, git_repo, store):
        indexer = CodebaseIndexer(store)
        idx = indexer.ensure_indexed("test/repo", str(git_repo))

        assert idx is not None
        assert idx.language == "python"
        assert idx.total_files > 0
        assert idx.total_lines > 0

        # Should find classes and functions
        symbol_names = {s.name for s in idx.symbols}
        assert "User" in symbol_names
        assert "Account" in symbol_names
        assert "handle_request" in symbol_names

    def test_detects_entry_points(self, git_repo, store):
        indexer = CodebaseIndexer(store)
        idx = indexer.ensure_indexed("test/repo", str(git_repo))

        assert "main.py" in idx.entry_points

    def test_detects_test_directories(self, git_repo, store):
        indexer = CodebaseIndexer(store)
        idx = indexer.ensure_indexed("test/repo", str(git_repo))

        assert "tests" in idx.test_directories

    def test_detects_modules(self, git_repo, store):
        indexer = CodebaseIndexer(store)
        idx = indexer.ensure_indexed("test/repo", str(git_repo))

        module_paths = {m.path for m in idx.modules}
        assert "src" in module_paths


class TestCaching:
    def test_second_call_uses_cache(self, git_repo, store):
        indexer = CodebaseIndexer(store)
        idx1 = indexer.ensure_indexed("test/repo", str(git_repo))
        idx2 = indexer.ensure_indexed("test/repo", str(git_repo))

        # Same SHA means cached
        assert idx1.commit_sha == idx2.commit_sha

    def test_new_commit_triggers_reindex(self, git_repo, store):
        indexer = CodebaseIndexer(store)
        idx1 = indexer.ensure_indexed("test/repo", str(git_repo))
        old_sha = idx1.commit_sha

        # Modify a file and commit
        (git_repo / "src" / "handler.py").write_text(
            'def handle_request_v2(req):\n    """Updated handler."""\n    return req\n'
        )
        _commit_all(git_repo, "update handler")

        idx2 = indexer.ensure_indexed("test/repo", str(git_repo))
        assert idx2.commit_sha != old_sha
        symbol_names = {s.name for s in idx2.symbols}
        assert "handle_request_v2" in symbol_names


class TestQueryIntegration:
    def test_query_returns_matching_symbols(self, git_repo, store):
        indexer = CodebaseIndexer(store)
        indexer.ensure_indexed("test/repo", str(git_repo))

        query = IndexQuery(store)
        result = query.query_for_prompt("test/repo", "User model")

        assert "User" in result
        assert "Codebase Overview" in result

    def test_overview_only(self, git_repo, store):
        indexer = CodebaseIndexer(store)
        indexer.ensure_indexed("test/repo", str(git_repo))

        query = IndexQuery(store)
        result = query.format_overview_only("test/repo")

        assert "python" in result
        assert "test/repo" in result


class TestSymbolCapping:
    def test_cap_prioritises_classes_over_methods(self, git_repo, store):
        """Tier 3 (class) should survive capping before tier 1 (method)."""
        indexer = CodebaseIndexer(store, max_symbols=3)
        idx = indexer.ensure_indexed("test/repo", str(git_repo))

        # The repo has 2 classes (User, Account) and 2+ functions —
        # with max_symbols=3, both classes must be kept
        names = {s.name for s in idx.symbols}
        assert "User" in names
        assert "Account" in names


class TestExcludePatterns:
    def test_excludes_pycache(self, git_repo, store):
        # Add a __pycache__ file
        (git_repo / "__pycache__").mkdir()
        (git_repo / "__pycache__" / "foo.cpython-311.pyc").write_text("bytecode")
        _commit_all(git_repo, "add pycache")

        indexer = CodebaseIndexer(
            store, exclude_patterns=["__pycache__/"]
        )
        idx = indexer.ensure_indexed("test/repo", str(git_repo))

        # __pycache__ files should not be in the index
        for sym in idx.symbols:
            assert "__pycache__" not in sym.file_path
