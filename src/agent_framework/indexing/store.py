"""Persistent storage for codebase indexes."""

import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from agent_framework.indexing.models import CodebaseIndex
from agent_framework.utils.atomic_io import atomic_write_model

logger = logging.getLogger(__name__)


class IndexStore:
    """Reads and writes CodebaseIndex JSON files under .agent-communication/indexes/."""

    def __init__(self, workspace: Path) -> None:
        self._base_dir = workspace / ".agent-communication" / "indexes"

    def _slug_dir(self, repo_slug: str) -> Path:
        safe = repo_slug.replace("/", "__")
        result = self._base_dir / safe
        # Prevent traversal outside base dir (defensive — slugs are admin-controlled)
        if not str(result.resolve()).startswith(str(self._base_dir.resolve())):
            raise ValueError(f"Invalid repo slug: {repo_slug!r}")
        return result

    def _index_path(self, repo_slug: str) -> Path:
        return self._slug_dir(repo_slug) / "codebase_index.json"

    def load(self, repo_slug: str) -> Optional[CodebaseIndex]:
        path = self._index_path(repo_slug)
        try:
            raw = path.read_text()
            return CodebaseIndex.model_validate_json(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Corrupt index for %s, ignoring: %s", repo_slug, exc)
            return None
        except OSError:
            return None

    def save(self, index: CodebaseIndex) -> None:
        path = self._index_path(index.repo_slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_model(path, index)

    def is_stale(self, repo_slug: str, current_commit_sha: str) -> bool:
        existing = self.load(repo_slug)
        if existing is None:
            return True
        return existing.commit_sha != current_commit_sha
