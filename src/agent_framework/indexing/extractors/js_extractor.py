"""JavaScript/TypeScript symbol extractor (M2)."""

from agent_framework.indexing.extractors.base import BaseExtractor
from agent_framework.indexing.models import SymbolEntry


class JSExtractor(BaseExtractor):

    def extract_symbols(self, file_path: str, source: str) -> list[SymbolEntry]:
        raise NotImplementedError
