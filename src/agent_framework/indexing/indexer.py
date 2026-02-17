"""Orchestrates full codebase indexing (M2)."""

from agent_framework.indexing.models import CodebaseIndex


class CodebaseIndexer:

    def ensure_indexed(self, repo_slug: str, repo_path: str) -> CodebaseIndex:
        raise NotImplementedError
