"""Orchestrates full codebase indexing."""

import fnmatch
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

from agent_framework.indexing.extractors import get_extractor_for_language
from agent_framework.indexing.extractors.base import MAX_FILE_SIZE, BaseExtractor
from agent_framework.indexing.models import CodebaseIndex, ModuleEntry, SymbolEntry, SymbolKind
from agent_framework.indexing.store import IndexStore
from agent_framework.utils.subprocess_utils import run_git_command

logger = logging.getLogger(__name__)

_EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".go": "go",
    ".rb": "ruby",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
}

# Reverse: language -> set of extensions
_LANGUAGE_EXTENSIONS: dict[str, set[str]] = {}
for _ext, _lang in _EXTENSION_MAP.items():
    _LANGUAGE_EXTENSIONS.setdefault(_lang, set()).add(_ext)

_ENTRY_POINT_NAMES: set[str] = {
    "main.py", "app.py", "manage.py", "wsgi.py", "asgi.py",
    "main.go",
    "config.ru", "Rakefile",
    "index.js", "index.ts", "server.js", "server.ts",
    "app.js", "app.ts",
}

_TEST_DIR_PARTS: set[str] = {"tests", "test", "spec", "__tests__"}

# Priority tiers for symbol capping
_PRIORITY_TIER: dict[str, int] = {
    SymbolKind.CLASS: 3,
    SymbolKind.STRUCT: 3,
    SymbolKind.INTERFACE: 3,
    SymbolKind.FUNCTION: 2,
    SymbolKind.MODULE: 2,
    SymbolKind.METHOD: 1,
}


class CodebaseIndexer:
    """Builds and caches a structural index of a repository."""

    def __init__(
        self,
        store: IndexStore,
        max_symbols: int = 500,
        exclude_patterns: list[str] | None = None,
        embedder=None,
    ) -> None:
        self._store = store
        self._max_symbols = max_symbols
        self._exclude_patterns = exclude_patterns or []
        self._embedder = embedder

    def ensure_indexed(
        self, repo_slug: str, repo_path: str
    ) -> Optional[CodebaseIndex]:
        """Return a (possibly cached) index. Never raises — returns None on failure."""
        try:
            return self._ensure_indexed_inner(repo_slug, Path(repo_path))
        except Exception:
            logger.warning("Indexing failed for %s, skipping", repo_slug, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_indexed_inner(
        self, repo_slug: str, repo_path: Path
    ) -> Optional[CodebaseIndex]:
        result = run_git_command(["rev-parse", "HEAD"], cwd=repo_path)
        head_sha = result.stdout.strip()

        # Single load — avoids double-deserialization through is_stale + load
        existing = self._store.load(repo_slug)
        if existing is not None and existing.commit_sha == head_sha:
            return existing

        # Compute changed files for incremental embedding updates
        changed_files = None
        deleted_files = None
        if existing is not None:
            changed_files, deleted_files = self._diff_files(
                repo_path, existing.commit_sha, head_sha
            )

        prior_sha = existing.commit_sha if existing else None

        index = self._build_index(repo_slug, repo_path, head_sha)
        if index is not None:
            self._store.save(index)
            self._try_embed_index(index, changed_files, deleted_files, prior_sha=prior_sha)
        return index

    def _build_index(
        self, repo_slug: str, repo_path: Path, commit_sha: str
    ) -> Optional[CodebaseIndex]:
        result = run_git_command(["ls-files"], cwd=repo_path)
        all_files = [f for f in result.stdout.splitlines() if f.strip()]

        tracked = [f for f in all_files if not self._is_excluded(f)]

        language = self._detect_language(tracked)
        if language is None:
            logger.info("No recognised language in %s", repo_slug)
            return None

        extractor = get_extractor_for_language(language)
        if extractor is None:
            logger.info("No extractor for %s in %s", language, repo_slug)
            return None

        lang_exts = _LANGUAGE_EXTENSIONS.get(language, set())
        source_files = [
            f for f in tracked if Path(f).suffix in lang_exts
        ]

        # Count lines across all tracked files so total_lines matches total_files
        total_lines = self._count_lines(repo_path, tracked)

        symbols: list[SymbolEntry] = []
        for rel in source_files:
            abs_path = repo_path / rel
            try:
                size = abs_path.stat().st_size
            except OSError:
                continue
            if size > MAX_FILE_SIZE:
                continue
            try:
                content = abs_path.read_text(errors="replace")
            except OSError:
                continue
            file_symbols = extractor.extract_symbols(rel, content)
            symbols.extend(file_symbols)

        symbols = self._cap_symbols(symbols)
        modules = self._build_modules(extractor, repo_path, source_files, language)
        entry_points = self._detect_entry_points(repo_path, tracked)
        test_dirs = self._detect_test_directories(tracked)

        return CodebaseIndex(
            repo_slug=repo_slug,
            commit_sha=commit_sha,
            language=language,
            total_files=len(tracked),
            total_lines=total_lines,
            modules=modules,
            symbols=symbols,
            entry_points=entry_points,
            test_directories=test_dirs,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_lines(repo_path: Path, files: list[str]) -> int:
        total = 0
        for rel in files:
            try:
                total += (repo_path / rel).read_text(errors="replace").count("\n")
            except OSError:
                continue
        return total

    def _is_excluded(self, path: str) -> bool:
        for pattern in self._exclude_patterns:
            if pattern.endswith("/"):
                # Directory prefix match
                if path.startswith(pattern) or f"/{pattern}" in f"/{path}":
                    return True
            elif fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(
                Path(path).name, pattern
            ):
                return True
        return False

    @staticmethod
    def _detect_language(files: list[str]) -> Optional[str]:
        counter: Counter[str] = Counter()
        for f in files:
            ext = Path(f).suffix
            lang = _EXTENSION_MAP.get(ext)
            if lang:
                counter[lang] += 1
        if not counter:
            return None
        return counter.most_common(1)[0][0]

    def _cap_symbols(self, symbols: list[SymbolEntry]) -> list[SymbolEntry]:
        if len(symbols) <= self._max_symbols:
            return symbols

        # Tag each symbol with its original position to restore order later
        indexed = list(enumerate(symbols))
        indexed.sort(key=lambda pair: _PRIORITY_TIER.get(pair[1].kind, 0), reverse=True)
        kept = indexed[: self._max_symbols]
        kept.sort(key=lambda pair: pair[0])
        return [sym for _, sym in kept]

    @staticmethod
    def _build_modules(
        extractor: BaseExtractor,
        repo_path: Path,
        source_files: list[str],
        language: str,
    ) -> list[ModuleEntry]:
        dirs: dict[str, list[str]] = {}
        for f in source_files:
            d = str(Path(f).parent) if Path(f).parent != Path(".") else "."
            dirs.setdefault(d, []).append(f)

        modules: list[ModuleEntry] = []
        for dir_rel, files in sorted(dirs.items()):
            abs_dir = str(repo_path / dir_rel) if dir_rel != "." else str(repo_path)
            mod = extractor.extract_module(abs_dir, files, language)
            # Override path to relative
            mod.path = dir_rel
            modules.append(mod)
        return modules

    @staticmethod
    def _detect_entry_points(repo_path: Path, files: list[str]) -> list[str]:
        found: list[str] = []
        for f in files:
            basename = Path(f).name
            if basename in _ENTRY_POINT_NAMES or f in _ENTRY_POINT_NAMES:
                # For main.go, verify it actually has func main()
                if basename == "main.go":
                    try:
                        content = (repo_path / f).read_text(errors="replace")[:4096]
                        if "func main()" not in content:
                            continue
                    except OSError:
                        continue
                found.append(f)
        return found

    @staticmethod
    def _detect_test_directories(files: list[str]) -> list[str]:
        test_dirs: set[str] = set()
        for f in files:
            parts = Path(f).parts
            for part in parts:
                if part in _TEST_DIR_PARTS:
                    # Build path up to and including the test dir
                    idx = parts.index(part)
                    test_dirs.add(str(Path(*parts[: idx + 1])))
                    break
            # Go convention: *_test.go files
            if f.endswith("_test.go"):
                parent = str(Path(f).parent)
                test_dirs.add(parent if parent != "." else ".")
        return sorted(test_dirs)

    # ------------------------------------------------------------------
    # Embedding support
    # ------------------------------------------------------------------

    @staticmethod
    def _diff_files(
        repo_path: Path, old_sha: str, new_sha: str
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """Compute changed and deleted files between two commits."""
        try:
            result = run_git_command(
                ["diff", "--name-status", old_sha, new_sha],
                cwd=repo_path,
            )
            changed: set[str] = set()
            deleted: set[str] = set()
            for line in result.stdout.strip().splitlines():
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                status = parts[0]
                if status.startswith("D"):
                    deleted.add(parts[1])
                elif status.startswith("R") or status.startswith("C"):
                    # Renames/copies: old_path\tnew_path — treat old as deleted, new as changed
                    if len(parts) >= 3:
                        deleted.add(parts[1])
                        changed.add(parts[2])
                else:
                    changed.add(parts[1])
            return changed, deleted
        except Exception:
            logger.debug("git diff failed between %s..%s", old_sha, new_sha, exc_info=True)
            return None, None

    def _try_embed_index(
        self,
        index: CodebaseIndex,
        changed_files: Optional[set[str]] = None,
        deleted_files: Optional[set[str]] = None,
        prior_sha: Optional[str] = None,
    ) -> None:
        """Best-effort embedding of the index. Failure only logs, never blocks."""
        if self._embedder is None:
            return

        try:
            from agent_framework.indexing.embeddings.vector_store import VectorStore

            store_path = self._store._slug_dir(index.repo_slug) / "lancedb"
            vector_store = VectorStore(store_path, dimensions=self._embedder.dimensions)

            # Incremental path: we have diff data AND the vector store matches the prior index
            can_incremental = (
                changed_files is not None
                and prior_sha is not None
                and not vector_store.is_stale(prior_sha)
            )
            if can_incremental:
                vector_store.update_incremental(
                    index, self._embedder,
                    changed_files, deleted_files or set(),
                )
            else:
                vector_store.rebuild(index, self._embedder)
        except Exception:
            logger.warning("Embedding index failed for %s, skipping", index.repo_slug, exc_info=True)
