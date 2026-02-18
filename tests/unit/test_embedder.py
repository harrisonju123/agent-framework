"""Tests for the Embedder wrapper — lazy load, prefixes, truncation, graceful degradation."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent_framework.indexing.embeddings.embedder import Embedder


def _make_embedder(dimensions: int = 256) -> Embedder:
    return Embedder(model_name="test-model", dimensions=dimensions)


class TestTriStateAvailability:
    def test_initial_state_is_none(self):
        e = _make_embedder()
        assert e._available is None

    def test_successful_load_sets_true(self):
        e = _make_embedder()
        mock_model = MagicMock()
        # Return 768-dim vectors
        mock_model.encode.return_value = np.random.randn(1, 768)
        with patch("sentence_transformers.SentenceTransformer",
                    return_value=mock_model) as mock_cls:
            # Trigger lazy load
            assert e._ensure_model() is True
            assert e._available is True
            mock_cls.assert_called_once_with("test-model", trust_remote_code=True)

    def test_failed_load_sets_false(self):
        e = _make_embedder()
        with patch("sentence_transformers.SentenceTransformer",
                    side_effect=RuntimeError("download failed")):
            assert e._ensure_model() is False
            assert e._available is False

    def test_no_retry_after_failure(self):
        e = _make_embedder()
        with patch("sentence_transformers.SentenceTransformer",
                    side_effect=RuntimeError("download failed")) as mock_cls:
            e._ensure_model()
            # Second call should not attempt import again
            result = e._ensure_model()
            assert result is False
            assert mock_cls.call_count == 1

    def test_no_reload_after_success(self):
        e = _make_embedder()
        mock_model = MagicMock()
        with patch("sentence_transformers.SentenceTransformer",
                    return_value=mock_model) as mock_cls:
            e._ensure_model()
            e._ensure_model()
            assert mock_cls.call_count == 1


class TestEmbedTexts:
    def _setup_embedder(self, dimensions=256, vector_dim=768):
        e = _make_embedder(dimensions=dimensions)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(3, vector_dim)
        e._model = mock_model
        e._available = True
        return e, mock_model

    def test_prefixes_with_search_document(self):
        e, mock_model = self._setup_embedder()
        texts = ["func foo", "class Bar", "module baz"]
        e.embed_texts(texts)

        call_args = mock_model.encode.call_args
        prefixed = call_args[0][0]
        assert all(t.startswith("search_document: ") for t in prefixed)
        assert prefixed[0] == "search_document: func foo"

    def test_show_progress_bar_false(self):
        e, mock_model = self._setup_embedder()
        e.embed_texts(["hello"])
        assert mock_model.encode.call_args[1]["show_progress_bar"] is False

    def test_returns_truncated_dimensions(self):
        e, mock_model = self._setup_embedder(dimensions=64)
        mock_model.encode.return_value = np.random.randn(2, 768)
        result = e.embed_texts(["a", "b"])
        assert result is not None
        assert len(result) == 2
        assert len(result[0]) == 64

    def test_empty_input_returns_empty(self):
        e, _ = self._setup_embedder()
        result = e.embed_texts([])
        assert result == []

    def test_returns_none_when_unavailable(self):
        e = _make_embedder()
        e._available = False
        assert e.embed_texts(["hello"]) is None

    def test_returns_none_on_encode_exception(self):
        e, mock_model = self._setup_embedder()
        mock_model.encode.side_effect = RuntimeError("OOM")
        result = e.embed_texts(["hello"])
        assert result is None


class TestEmbedQuery:
    def test_prefixes_with_search_query(self):
        e = _make_embedder()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 768)
        e._model = mock_model
        e._available = True

        e.embed_query("improve user onboarding")
        call_args = mock_model.encode.call_args
        assert call_args[0][0] == ["search_query: improve user onboarding"]

    def test_returns_single_vector(self):
        e = _make_embedder(dimensions=128)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 768)
        e._model = mock_model
        e._available = True

        result = e.embed_query("test query")
        assert result is not None
        assert len(result) == 128

    def test_returns_none_when_unavailable(self):
        e = _make_embedder()
        e._available = False
        assert e.embed_query("hello") is None

    def test_returns_none_on_encode_exception(self):
        e = _make_embedder()
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("OOM")
        e._model = mock_model
        e._available = True
        assert e.embed_query("hello") is None


class TestMatryoshkaTruncation:
    def test_truncation_and_renormalization(self):
        """Truncated vectors must be re-normalized to unit length."""
        e = _make_embedder(dimensions=64)
        # Create a known vector
        raw = np.random.randn(1, 768)
        e._model = MagicMock()
        e._available = True

        result = e._truncate_and_normalize(raw)
        assert result.shape == (1, 64)
        # Check unit length (L2 norm ≈ 1.0)
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-6

    def test_batch_renormalization(self):
        e = _make_embedder(dimensions=32)
        raw = np.random.randn(5, 768)

        result = e._truncate_and_normalize(raw)
        assert result.shape == (5, 32)
        for row in result:
            assert abs(np.linalg.norm(row) - 1.0) < 1e-6

    def test_zero_vector_handled(self):
        """Zero vectors should not cause division-by-zero."""
        e = _make_embedder(dimensions=4)
        raw = np.zeros((1, 768))
        result = e._truncate_and_normalize(raw)
        # Should not be NaN
        assert not np.any(np.isnan(result))
