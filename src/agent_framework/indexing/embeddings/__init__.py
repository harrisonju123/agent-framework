"""Embedding-based semantic search for codebase indexes."""

try:
    import sentence_transformers  # noqa: F401
    import lancedb  # noqa: F401
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
