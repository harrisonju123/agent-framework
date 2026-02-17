"""Persistent storage for codebase indexes (M2)."""

from typing import Optional

from agent_framework.indexing.models import CodebaseIndex


class IndexStore:

    def load(self, repo_slug: str) -> Optional[CodebaseIndex]:
        raise NotImplementedError

    def save(self, repo_slug: str, index: CodebaseIndex) -> None:
        raise NotImplementedError
