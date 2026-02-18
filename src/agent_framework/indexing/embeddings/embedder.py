"""Wraps SentenceTransformer for embedding code symbols and queries."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"


class Embedder:
    """Lazy-loading embedding model with Matryoshka dimension truncation."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        dimensions: int = 256,
    ) -> None:
        self._model_name = model_name
        self._dimensions = dimensions
        self._model = None
        # Tri-state: None = untried, True = loaded, False = failed permanently
        self._available: Optional[bool] = None

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _ensure_model(self) -> bool:
        """Load model on first call. Returns False on failure, never retries."""
        if self._available is True:
            return True
        if self._available is False:
            return False

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self._model_name,
                trust_remote_code=True,
            )
            self._available = True
            return True
        except Exception:
            logger.warning("Failed to load embedding model %s", self._model_name, exc_info=True)
            self._available = False
            return False

    def _truncate_and_normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Matryoshka truncation to target dimensions + L2 re-normalization."""
        truncated = vectors[:, :self._dimensions]
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return truncated / norms

    def embed_texts(self, texts: list[str]) -> Optional[list[list[float]]]:
        """Batch embed document texts with 'search_document: ' prefix."""
        if not self._ensure_model():
            return None
        if not texts:
            return []

        prefixed = [f"search_document: {t}" for t in texts]
        try:
            raw = self._model.encode(prefixed, show_progress_bar=False)
            normalized = self._truncate_and_normalize(np.array(raw))
            return normalized.tolist()
        except Exception:
            logger.warning("embed_texts failed for %d texts", len(texts), exc_info=True)
            return None

    def embed_query(self, text: str) -> Optional[list[float]]:
        """Embed a single query with 'search_query: ' prefix."""
        if not self._ensure_model():
            return None

        try:
            raw = self._model.encode(
                [f"search_query: {text}"],
                show_progress_bar=False,
            )
            normalized = self._truncate_and_normalize(np.array(raw))
            return normalized[0].tolist()
        except Exception:
            logger.warning("embed_query failed", exc_info=True)
            return None
