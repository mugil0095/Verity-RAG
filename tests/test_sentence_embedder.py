"""
Tests SentenceTransformerEmbedder's WRAPPER LOGIC using a mocked
SentenceTransformer: dimension reporting, empty-input handling, and that
normalize_embeddings=True is actually requested (cosine_sim_matrix assumes
pre-normalized input -- see embedding.py). None of this needs network access
or a real model download, so it runs the same in CI as anywhere else.

This does NOT verify the real model loads and produces sensible embeddings
end to end -- see test_sentence_embedder_live.py for that.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("sentence_transformers")


def test_wrapper_reports_model_dimension():
    fake_model = MagicMock()
    fake_model.get_sentence_embedding_dimension.return_value = 384
    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
        from src.verityrag.embedding import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder("fake-model")
    assert emb.dim == 384


def test_wrapper_requests_normalized_embeddings():
    fake_model = MagicMock()
    fake_model.get_sentence_embedding_dimension.return_value = 4
    fake_model.encode.return_value = np.array([[0.5, 0.5, 0.5, 0.5]])
    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
        from src.verityrag.embedding import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder("fake-model")
        result = emb.embed(["some text"])

    fake_model.encode.assert_called_once()
    _, kwargs = fake_model.encode.call_args
    # required: every downstream cosine_sim_matrix() call assumes this
    assert kwargs.get("normalize_embeddings") is True
    assert result.shape == (1, 4)


def test_wrapper_handles_empty_input_without_calling_model():
    fake_model = MagicMock()
    fake_model.get_sentence_embedding_dimension.return_value = 4
    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
        from src.verityrag.embedding import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder("fake-model")
        result = emb.embed([])

    assert result.shape == (0, 4)
    fake_model.encode.assert_not_called()