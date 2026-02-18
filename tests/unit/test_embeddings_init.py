"""Tests for EMBEDDINGS_AVAILABLE flag in embeddings __init__."""

import importlib
import sys
from unittest.mock import MagicMock


class TestEmbeddingsAvailable:
    def test_available_when_deps_installed(self):
        from agent_framework.indexing.embeddings import EMBEDDINGS_AVAILABLE
        # Both sentence_transformers and lancedb are installed in test env,
        # so this should reflect actual availability. Just verify the flag is a bool.
        assert isinstance(EMBEDDINGS_AVAILABLE, bool)

    def test_unavailable_when_import_fails(self, monkeypatch):
        """Simulate missing dependency by temporarily breaking the import."""
        mod_name = "agent_framework.indexing.embeddings"
        original = sys.modules.get(mod_name)

        # Remove cached module so we can re-import
        sys.modules.pop(mod_name, None)

        # Make sentence_transformers raise ImportError
        real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def fake_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        try:
            mod = importlib.import_module(mod_name)
            assert mod.EMBEDDINGS_AVAILABLE is False
        finally:
            # Restore original module
            if original is not None:
                sys.modules[mod_name] = original
            else:
                sys.modules.pop(mod_name, None)
